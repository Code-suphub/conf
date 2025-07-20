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
from matplotlib import pyplot as plt
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

"""
目前rho2= 500 , rho1 = 5是最优解
"""
rho, rho2,alpha = common.get_rho()
log2 = False
# alpha = 0.1
alpha = 1
# alpha = 10
# rho2 = 10
# rho2 = 100
# rho2 = 1000
# rho2 = 10000

# rho = 3
# rho = 4
# rho = 5
# rho = 6
# rho = 7
rho = 8
# rho = 9

# rho2 = 50
# rho2 = 200
rho2 = 500
# rho2 = 2000
# rho2 = 5000
# rho2 = 20000
# rho2 = 50000


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

    res = [] # 最终保存结果的地方

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
    cnt = 0
    sumT=0
    cutlay_lst = [0]*100 # 0: 4435, 1: 7047, 2: 7525, 3: 7633, 4: 7640
    t1 = time.time()
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
    algo.alpha = alpha
    algo.cutlayer_lst = cutlay_lst[:]

    fld,sld = algo.cal_delay()

    local_optim = 0

    ut_value = max(fld, sld) - rho*(sum(sl_lst)*(sum(sl_lst)-1)) #  归一化求解
    total_delay = max(fld, sld)
    sl_num_lst = []
    if log2:
        with open(f"../save/output/conference/local_cnt[1]_user30_rho1[{rho}]_rho2[{rho2}]_alpha[{alpha}].csv", 'w') as f:
            pass
    ut_list = []
    algo.batch_size_init()

    ut___lst = []

    while True:
        # 这个 for 循环是为了通过轮询的方式解决分别解决P1 和 P2
        local_optim = 0
        # TODO 这里的G目前降低了，加速训练
        G = 200
        ut_G_lst = []

        ut_lst = []
        #
        algo.batch_size_decision(log=False,log2=log2)
        # TODO 对于batch的方式，直接 全是SL ，由于计算时延较小
        ut_value, total_delay = algo.cal_old_ut()  # 归一化求解
        ut_value__, max_delay = algo.cal_ut()

        # ut___lst.append(algo.round())

        if len(ut_list)==0:
            ut_list.append(ut_value__)

        for local_optim in range(G):
            old_algo = copy.deepcopy(algo)
            fl_lst, sl_lst = common.generate_new_lst(algo.fl_lst[:], algo.sl_lst[:], args)
            algo.update_partition(fl_lst[:], sl_lst[:])
            ind = 0
            b0 = algo.binary_b0(True, True,cutlay_lst=cutlay_lst)
            # ut_new_value, new_delay = algo.cal_old_ut()  # 归一化求解
            # TODO 如果这里计算ut不适用batch的值，会导致坐标轮询失效
            ut_new_value, new_delay = algo.cal_ut()  # 归一化求解

            # print("a: ",a,"  b: ",b,"  c: ",c)
            ut_dif = ut_new_value - ut_value
            epsilon = common.cal_epsilon(ut_dif)
            if random.random() > epsilon:
                algo = copy.deepcopy(old_algo)
            else:
                ut_value = ut_new_value
                total_delay = new_delay

            ut_lst.append(ut_value)

        ut_new_value, _ = algo.cal_ut()
        if log2:
            print("坐标轮询次数：",cnt)
            if cnt >= 100:
                break
        else:
            if cnt >= 50:
                break
        ut_value = ut_new_value
        ut_list.append(ut_value)
        cnt += 1
        ut_lst.append(ut_value)
        # 1/0
    ut_new_value, _ = algo.cal_ut()
    print(f"rho1:\t{rho} \t rho2:\t{rho2} \t ut_value:\t{ut_new_value}")