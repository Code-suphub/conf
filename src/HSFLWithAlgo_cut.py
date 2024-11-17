#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import json
import math
import os
import copy
import random
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
import common
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, shard_num_generate, solve, mySolve, mySolve3, mySolve2
from param_cal import cal_model_param, cal_uplink_rate, cal_model_activation, cal_model_flops
from random_generate import compute_capacity_rand_generate
from scipy.special import lambertw
from alog import Algo


rho, rho2 = common.get_rho()

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    seed = 124
    sigma = 0.0075
    try:
        from pytorch_lightning import seed_everything

        seed_everything(seed)
    except:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    args = args_parser()
    exp_details(args)
    args.epochs = 150
    div = 1  # 7 和 17
    print("div: ",div)

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
    print_every = 2
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
    res = []

    file_name,args = common.get_file_name(args,"HSFLAlgoCut")

    for epoch in tqdm(range(args.epochs)):
        compute_list = compute_capacity_rand_generate(args.num_users)  # 获取每个用户的计算能力
        capacity, signal_cap,Bandwidth,noise,pku = cal_uplink_rate(args.num_users)  # 获取每个用户的噪声和信号功率
        local_weights, local_losses = [0] * args.num_users, [0] * args.num_users
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        decisionLst = []
        while True:
            action_lst = [random.randint(0, 1) for _ in range(args.num_users)]
            fl_lst = [1 if action_lst[i] == 1 else 0 for i in range(args.num_users)]
            sl_lst = [1 if action_lst[i] == 0 else 0 for i in range(args.num_users)]  # 随机初始化两种学习方式的客户端
            if sum(fl_lst) == 0:
                continue
            else:
                break
        algo = Algo(fl_lst,sl_lst,capacity,Bandwidth,signal_cap,model_param,activations,user_groups,
                    args,flops,compute_list,sample_size,pku,rho2 = 10)

        fld,sld = algo.cal_delay()

        local_optim = 0

        ut_value = max(fld, sld) - (sum(sl_lst)*(sum(sl_lst)-1))/rho #  归一化求解
        total_delay = max(fld, sld)
        ut_lst = []
        G = 250
        # for local_optim in range(G):
        for local_optim in range(G):
            old = (fl_lst[:],sl_lst[:])
            fl_lst,sl_lst = common.generate_new_lst(fl_lst,sl_lst,args)
            algo.update_partition(fl_lst,sl_lst)
            ind = 0
            algo.cutlayer_lst = [2 for i in range(len(fl_lst))]
            # b0 = algo.binary_b0(False,True)
            b0 = algo.binary_b0(True,False)
            fld,sld = algo.cal_delay()

            # a = max(fld, sld)/sldUp
            # b = (sum(sl_lst)*(sum(sl_lst)-1))/(args.num_users*(args.num_users-1))
            a = max(fld, sld)
            b = (sum(sl_lst)*(sum(sl_lst)-1))/rho
            delay = max(fld, sld)
            # ut_value_new = max(fld, sld)  - (sum(sl_lst)*(sum(sl_lst)-1))/(args.num_users*(args.num_users-1)) #  归一化求解#  归一化求解
            ut_value_new = a - b #  归一化求解
            ut_dif = ut_value_new - ut_value
            epsilon = common.cal_epsilon(ut_dif)
            if random.random()>epsilon:
                fl_lst,sl_lst = old  # 回滚
            else:
                ut_value = ut_value_new
                total_delay = delay
            # local_optim += 1
            # fld = [t + j for t, j in zip(fltd, fljd)]
            # # sld = [t + j for t, j in zip(sltd, sljd)]
            # decisionLst.append([fl_lst[:], sl_lst[:], "slCLient: " + str(sum(sl_lst)), "flCLient: " + str(sum(fl_lst)),
            #                     max(max(fld), sld)])
            ut_lst.append(float(ut_value))
        with open(f"../save/conferenceRes/ut_sigma{sigma}_1",'w') as f:
            json.dump(ut_lst,f)
        # decisionLst.sort(key=lambda x: x[-1])
        # [print(l) for l in decisionLst]
        # for dec in decisionLst:
        #     if int(dec[2][-2:])>=8:
        #         fl_lst, sl_lst = dec[0], dec[1]
        #         res.append([dec[2],dec[-1]])
        #         break
        res.append([sum(sl_lst), total_delay])
        # continue
        ind = 0
        for idx, a in enumerate(fl_lst):
            if a == 1:
                print(idx, '----------------', len(user_groups[idx]), '--------------', len(user_groups))
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx])
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch, local_losses=local_losses,
                    local_weights=local_weights)
                local_weights[ind] = copy.deepcopy(w)
                local_losses[ind] = copy.deepcopy(loss)
                ind += 1

        for idx, a in enumerate(sl_lst):
            if a == 1:
                print(idx, '----------------', len(user_groups[idx]), '--------------', len(user_groups))
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx])
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch, local_weights=local_weights,
                    local_losses=local_losses)
                local_weights[ind] = copy.deepcopy(w)
                local_losses[ind] = copy.deepcopy(loss)
                global_model.load_state_dict(w)
                ind += 1

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
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
        try:
            res[epoch].append(train_accuracy[-1])
            res[epoch].append(np.mean(np.array(train_loss)))
            res[epoch].append(train_loss[-1])
            res[epoch].append(sum(list_loss) / len(list_loss))
        except:
            time.sleep(1000000)
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
    # # Test inference after completion of training
    print(res)
    # file_name = '../save/conferenceRes/20HSFL{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_lr[{}].csv'. \
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs,args.lr)

    # file_name = '../save/conferenceRes/10HSFL{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_lr[{}].csv'. \
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs, args.lr)
    # file_name = '../save/conferenceRes/0.0130HSFLAlgo{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_lr[{}]_div{}.csv'. \
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs, args.lr,div)

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

    # # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')
    #
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
