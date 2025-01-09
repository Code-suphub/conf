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

from options import args_parser
import common
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, shard_num_generate, solve, mySolve, mySolve3, mySolve2
from param_cal import cal_model_param, cal_uplink_rate, cal_model_activation, cal_model_flops
from random_generate import compute_capacity_rand_generate
from scipy.special import lambertw
from alog import Algo

rho, rho2,alpha = common.get_rho()
# alpha = 0.1
# alpha = 1
# alpha = 10
# rho2 = 0.0001
# rho2 = 0.001
# rho2 = 0.01
# rho2 = 0.1
# rho2 = 1
# rho2 = 10
# rho2 = 100
# rho2 = 1000
rho2 = 10000
# rho = 0.0001
# rho = 0.001
rho = 0.01
# rho = 0.1
# rho = 1
# rho = 10
# rho = 100
# rho = 1000
# rho = 10000
# rho = 100000
# rho = 1000000
# rho = 10000000
# rho = 100000000

if __name__ == '__main__':
    print("rho:",rho,"  rho2:",rho2, "   alpha:", alpha)
    # for rho in common.rho_lst:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    sigma = 0.00075

    common.train_seed()
    args = args_parser()
    exp_details(args)
    args.epochs = 1000
    div = 1  # 7 和 17
    print("div: ",div)

    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    user_groups = shard_num_generate(np.array(train_dataset.targets), alpha ,args.num_users)
    # BUILD MODEL
    global_model,tempModel = common.model_get(args,train_dataset)

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

    file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{rho}]_rho2[{rho2}]_alpha[{alpha}].csv"
    res = ""
    cnt = 0

    for epoch in tqdm(range(args.epochs)):
        compute_list = compute_capacity_rand_generate(args.num_users)  # 获取每个用户的计算能力
        capacity, signal_cap,Bandwidth,noise,pku = cal_uplink_rate(args.num_users)  # 获取每个用户的噪声和信号功率
        local_weights, local_losses = [0] * args.num_users, [0] * args.num_users

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
        algo = Algo(fl_lst[:],sl_lst[:],capacity,Bandwidth,signal_cap,model_param,activations,user_groups,
                    args,flops,compute_list,sample_size,pku,rho2 =rho2)
        algo.rho = rho
        algo.rho2 = rho2

        fld,sld = algo.cal_delay()

        local_optim = 0

        ut_value = max(fld, sld) - rho*(sum(sl_lst)*(sum(sl_lst)-1)) #  归一化求解
        total_delay = max(fld, sld)
        ut_lst = []
        ut_lst_lst = []
        G = 200
        with open("../save/output/conference/local_cnt[1]_user30_rho1[{rho}]_rho2[{rho2}]_alpha[{alpha}].csv", 'a') as f:
            pass
        while True:
            # 这个 for 循环是为了通过轮询的方式解决分别解决P1 和 P2
            local_optim = 0

            ut_value, total_delay = algo.cal_ut()  # 归一化求解
            # TODO 这里的G目前降低了，加速训练
            G = 200
            ut_G_lst = []

            algo.batch_size_decision(log=False)
            # TODO 对于batch的方式，直接 全是SL ，由于计算时延较小
            for local_optim in range(G):

                fl_lst, sl_lst = common.generate_new_lst(fl_lst, sl_lst, args)

                old_algo = copy.deepcopy(algo)
                algo.update_partition(fl_lst[:], sl_lst[:])
                ind = 0
                b0 = algo.binary_b0(True, True)
                ut_new_value, new_delay = algo.cal_ut()  # 归一化求解

                # print("a: ",a,"  b: ",b,"  c: ",c)
                ut_dif = ut_new_value - ut_value
                epsilon = common.cal_epsilon(ut_dif)
                if random.random() > epsilon:
                    algo = copy.deepcopy(old_algo)
                else:
                    ut_value = ut_new_value
                    total_delay = new_delay

            ut_new_value, _ = algo.cal_ut()
            print("坐标轮询次数：",cnt)
            # if cnt >= 15:
            if cnt >= 100:
                break
            ut_value = ut_new_value
            cnt += 1
            ut_lst.append(ut_value)

        sample_data = []
        for idx in range(len(user_groups)):
            # TODO 对于SL排序后一次递增最小的，直到不满足SUM（SL_DELAY) <= TAU
            sample_data.append(
                random.sample(user_groups[idx][:int(0.8 * len(user_groups[idx]))], int(algo.batch_size_lst[idx])))
        for idx, a in enumerate(fl_lst):
            if a == 1:
                print("FL：ind:", idx, '---------------- batch_size_lst: ', algo.batch_size_lst[idx], '------------',
                      len(sample_data[idx]), '--------------', len(user_groups))
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=sample_data[idx], sign=True,
                                          validData=user_groups[idx][
                                                    int(0.8 * len(user_groups[idx])):int(0.9 * len(user_groups[idx]))],
                                          testData=user_groups[idx][
                                                   int(0.9 * len(user_groups[idx])):int(len(user_groups[idx]))])
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch, local_losses=local_losses,
                    local_weights=local_weights)
                local_weights[ind] = copy.deepcopy(w)
                local_losses[ind] = copy.deepcopy(loss)
                ind += 1

        for idx, a in enumerate(sl_lst):
            if a == 1:
                print("SL：   ", idx, '----------------', algo.batch_size_lst[idx], '------------',
                      len(sample_data[idx]), '--------------', len(user_groups))
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=sample_data[idx], sign=True,
                                          validData=user_groups[idx][
                                                    int(0.8 * len(user_groups[idx])):int(0.9 * len(user_groups[idx]))],
                                          testData=user_groups[idx][
                                                   int(0.9 * len(user_groups[idx])):int(len(user_groups[idx]))])
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

        res+= f"{sum(algo.sl_lst)},{total_delay},{train_accuracy[-1]},{sum(algo.batch_size_lst)}\n"
        with open(file_name,'w') as f:
            f.write(res)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))