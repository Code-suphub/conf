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
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details, shard_num_generate, solve, mySolve, mySolve3, mySolve2
from param_cal import cal_model_param, cal_uplink_rate, cal_model_activation, cal_model_flops
from random_generate import compute_capacity_rand_generate
from scipy.special import lambertw
from alog import Algo
import common

rho, rho2,alpha = common.get_rho()
# rho = 500
# rho2 = 10

if __name__ == '__main__':
    for sigma in common.sigma_lst:
        if True:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            start_time = time.time()

            # define paths
            path_project = os.path.abspath('..')

            common.train_seed()

            args = args_parser()
            exp_details(args)
            args.epochs = 150
            div = 1  # 7 和 17
            print("div: ", div)

            device = 'cuda' if args.gpu else 'cpu'

            # load dataset and user groups
            train_dataset, test_dataset, user_groups = get_dataset(args)
            user_groups = shard_num_generate(np.array(train_dataset.targets), alpha ,args.num_users)
            # BUILD MODEL
            global_model, tempModel = common.model_get(args, train_dataset)

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

        """
        每一轮训练开始阶段，随机初始化每个客户端的训练行为a_k,t,
            训练开始之后，通过二分搜索找到b_0,t,(先假定FL共同的带宽用来传输所有的数据，即FL的时延由计算能力最慢的设备决定，同时SL未优化分割层在第2层/第一层）
                对于SL用户，得到了b_0,t  ,那么只需要优化分割层
                对于FL 用户，得到了b_0,t 只需要优化每个FL用户的带宽分配  训练开始之后，通过gibbs算法结合二分时间搜索获取带宽和x_k
            然后通过gibbs进行用户训练行为的更改
        """

        res = []  # 最终保存结果的地方

        file_name, args = common.get_file_name(args, "AlgoWithBatch", f"_rho2[{rho2}]",alpha = alpha)

        td = ""
        sumT = 0
        cnt = 0
        for epoch in tqdm(range(args.epochs)):
            t1 = time.time()
            fl_lst = [0]*args.num_users
            sl_lst = [1]*args.num_users
            compute_list = compute_capacity_rand_generate(args.num_users)  # 获取每个用户的计算能力
            capacity, signal_cap,Bandwidth,noise,pku = cal_uplink_rate(args.num_users)  # 获取每个用户的噪声和信号功率
            local_weights, local_losses = [0] * args.num_users, [0] * args.num_users
            print(f'\n | Global Training Round : {epoch + 1} |\n')

            global_model.train()
            decisionLst = []

            # while True:
            #     # 进行随机初始化fl和sl的，尽量不出现全是sl的或者全是fl的
            #     action_lst = [random.randint(0, 1) for _ in range(args.num_users)]
            #     fl_lst = [1 if action_lst[i] == 1 else 0 for i in range(args.num_users)]
            #     sl_lst = [1 if action_lst[i] == 0 else 0 for i in range(args.num_users)]  # 随机初始化两种学习方式的客户端
            #     if sum(fl_lst) == 0 or sum(sl_lst) == 0:
            #         continue
            #     else:
            #         break
            fl_lst = [0 for i in range(args.num_users)]
            sl_lst = [1 for i in range(args.num_users)]  # 随机初始化两种学习方式的客户端
            algo = Algo(fl_lst, sl_lst, capacity, Bandwidth, signal_cap, model_param, activations, user_groups,
                        args, flops, compute_list, sample_size, pku,rho2)
            ut_lst = []
            # 这个 for 循环是为了通过轮询的方式解决分别解决P1 和 P2
            fld, sld = algo.cal_delay(algo.batch_size_lst)

            local_optim = 0

            ut_value = max(fld, sld) - (sum(sl_lst) * (sum(sl_lst) - 1)) / rho + sum([1/xi/rho2 for xi in algo.batch_size_lst])   # 归一化求解
            total_delay = max(fld, sld)
            G = 1000
            ut_G_lst = []

            algo.batch_size_decision()
            # TODO 对于batch的方式，直接 全是SL ，由于计算时延较小
            for local_optim in tqdm(range(G)):
                old = (fl_lst[:], sl_lst[:])
                fl_lst,sl_lst = common.generate_new_lst(fl_lst,sl_lst,args)
                algo.update_partition(fl_lst, sl_lst)
                ind = 0
                b0 = algo.binary_b0(True, True)
                fld, sld = algo.cal_delay(algo.batch_size_lst)
                # print(fld, sld)
                # ind+=1
                # if ind>=gap_end:
                #     print(fld,sld)
                #     break

                # a = max(fld, sld)/sldUp
                # b = (sum(sl_lst)*(sum(sl_lst)-1))/(args.num_users*(args.num_users-1))
                a = max(fld, sld)
                b = (sum(sl_lst) * (sum(sl_lst) - 1)) / rho
                delay = max(fld, sld)
                # ut_value_new = max(fld, sld)  - (sum(sl_lst)*(sum(sl_lst)-1))/(args.num_users*(args.num_users-1)) #  归一化求解#  归一化求解
                c = sum([1/xi/rho2 for xi in algo.batch_size_lst])
                ut_value_new = a - b + c  # 归一化求解
                # print("a: ",a,"  b: ",b,"  c: ",c)
                ut_dif = ut_value_new - ut_value
                epsilon = common.cal_epsilon(ut_dif,sigma = sigma)
                if random.random() > epsilon:
                    fl_lst, sl_lst = old  # 回滚
                else:
                    ut_value = ut_value_new
                    total_delay = delay
                ut_G_lst.append(ut_value)

            with open("../save/output/conference/cmpResult/sigma/inner/sigma_"+str(sigma)+".csv",'w') as f:
                f.write(",".join([str(i) for i in ut_G_lst]))
            break