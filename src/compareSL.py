import json
import math
import os
import copy
import random
import threading
import time
import pickle
import numpy as np
from tqdm import tqdm

from options import args_parser
from common import *
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, shard_num_generate
from param_cal import cal_model_param, cal_uplink_rate, cal_model_activation, cal_model_flops
from random_generate import compute_capacity_rand_generate
from scipy.special import lambertw
from alog import Algo

if __name__ == '__main__':
    if True:
        # 基本参数设定
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        start_time = time.time()
        seed = 124
        try:
            from pytorch_lightning import seed_everything
            seed_everything(seed)
        except:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # define paths
        path_project = os.path.abspath('..')

        args = args_parser()
        exp_details(args)
        args.epochs = 100

        # if args.gpu:
        #     torch.cuda.set_device(args.gpu)
        device = 'cuda' if args.gpu else 'cpu'
        # device = 'cpu'

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = get_dataset(args)
        user_groups = shard_num_generate(args.num_users,len(train_dataset))

        # BUILD MODEL
        if args.model == 'cnn':
            # Convolutional neural netork
            if args.dataset == 'mnist':
                tempModel = CNNMnist(args=args)
                global_model = CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                tempModel = CNNFashion_Mnist(args=args)
                global_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar':
                tempModel = CNNCifar(args=args)
                global_model = CNNCifar(args=args)

        elif args.model == 'mlp':
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=64,
                                   dim_out=args.num_classes)
        else:
            exit('Error: unrecognized model')

        # Set the model to train and send it to device.
        print(global_model)

        # copy weights
        global_weights = global_model.state_dict()

        # Training
        train_loss, train_accuracy = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 1
        val_loss_pre, counter = 0, 0

    """
    每一轮训练开始阶段，随机初始化每个客户端的训练行为a_k,t,
        训练开始之后，通过二分搜索找到b_0,t,(先假定FL共同的带宽用来传输所有的数据，即FL的时延由计算能力最慢的设备决定，同时SL未优化分割层在第2层/第一层）
            对于SL用户，得到了b_0,t  ,那么只需要优化分割层
            对于FL 用户，得到了b_0,t 只需要优化每个FL用户的带宽分配  训练开始之后，通过gibbs算法结合二分时间搜索获取带宽和x_k
        然后通过gibbs进行用户训练行为的更改
    """
    sample = train_dataset[0][0]
    sample_size = sample.shape[0] * sample.shape[1] * sample.shape[2]
    with open("tempDate/activations", "r") as f:
        activations = json.load(f)
    with open("tempDate/flops", "r") as f:
        flops = json.load(f)
    with open("tempDate/model_param", "r") as f:
        model_param = json.load(f)
    global_model.to(device)
    global_model.train()
    res =[] # 最后的结果保存，【
    cut_layer = 0
    rho2 = 10

    file_name,args = get_file_name(args,"SL")

    for epoch in tqdm(range(args.epochs)):
        sl_lst = [1]*args.num_users
        fl_lst = [0]*args.num_users
        compute_list = compute_capacity_rand_generate(args.num_users)  # 获取每个用户的计算能力
        capacity,signal_cap,Bandwidth,noise,pku = cal_uplink_rate(args.num_users)  # 获取每个用户的噪声和信号功率
        local_weights,local_losses = [0]*args.num_users , [0]*args.num_users
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        algo = Algo(fl_lst, sl_lst, capacity, Bandwidth, signal_cap, model_param, activations, user_groups,
                    args, flops, compute_list, sample_size, pku, rho2)
        algo.cutlayer_lst = [cut_layer for _ in range(len(fl_lst))]

        # slr = [1 if sl_lst[i] == 1 else 0 for i in range(len(sl_lst))]
        # sluc = [(Bandwidth* slr[i] * np.log2(1 + pku * signal_cap[i] / Bandwidth* slr[i] / noise) ).item()for i in range(len(sl_lst))]
        # sldc = [(Bandwidth* slr[i] * np.log2(1 + pku * signal_cap[i] / Bandwidth* slr[i] / noise) ).item() for i in
        #         range(args.num_users)]
        # # TODO 服务器的下行速率没有
        # sltd = [model_param[cut_layer] / uc + model_param[cut_layer] / dc
        #               + activations[cut_layer][-1] * len(user_groups[i]) / uc + activations[cut_layer][-1] * len(user_groups[i]) / dc
        #               if uc > 0 else 0 for i, (uc, dc)
        #               in enumerate(zip(sluc, sldc))]  # 每个客户端的通信时延,sl还有激活值的时延
        # # fltd2 = [model_param[-1] / c if c > 0 else 0 for c in flc2]
        # aaa = model_param[1]
        # bbb = activations[1][-1] * len(user_groups[0])
        # aaa = model_param[2]
        # bbb = activations[2][-1] * len(user_groups[0])
        # aaa = model_param[3]
        # bbb = activations[3][-1] * len(user_groups[0])
        # sljd = [(flops[cut_layer] / compute_list[i]) + (flops[-1] - flops[cut_layer]) / compute_list[-1]
        #                 if sl_lst[i] == 1 else 0 for i in range(len(sl_lst))]  # 每个客户端的计算时延 , sl 包含本地计算时延和服务器计算时延
        # sld = [t + j for t, j in zip(sltd, sljd)]  # 每个客户端的时延
        # sld =sum(sld)  # sl 是取总和，fl是取最大值
        batch_size = [len(i) for i in user_groups]
        fld,sld = algo.cal_delay(batch_size)
        res.append([len(sl_lst),sld])  # 保留sl用户的数量和本轮的时延
        ind=0
        for idx,a in enumerate(sl_lst):
            if a==1:
                # print(idx, '----------------', len(user_groups[idx]), '--------------', len(user_groups))
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx])

                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch,local_weights=local_weights,local_losses=local_losses)
                local_weights[ind] = copy.deepcopy(w)
                local_losses[ind] = copy.deepcopy(loss)
                global_model.load_state_dict(w)
                ind+=1
        # local_model = [LocalUpdate(args=args, dataset=train_dataset,
        #                            idxs=user_groups[idx]) for idx in sl_lst]
        # threads = [threading.Thread(target=client.update_weights, args=(
        #     copy.deepcopy(global_model), epoch, local_weights, local_losses, i)) for i, client in
        #            enumerate(local_model)]
        # [t.start() for t in threads]
        # [t.join() for t in threads]
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        ind=1
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
        res[epoch].append(train_accuracy[-1])
        res[epoch].append(np.mean(np.array(train_loss)))
        res[epoch].append(train_loss[-1])
        res[epoch].append(sum(list_loss) / len(list_loss))
        try:
            for i in range(len(res)):
                for j in range(len(res[0])):
                    res[i][j] = float(res[i][j])
            with open(file_name, 'w') as f:
                json.dump(res, f)
        except:
            print("json 保存字符串")
            with open(file_name, 'w') as f:
                json.dump(str(res), f)
    # Test inference after completion of training
    print(res)
    # file_name = '../save/conferenceRes/10SL{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_lr[{}].csv'. \
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs,args.lr)

    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))