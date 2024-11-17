#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import math
import random

# import matplotlib.pyplot as plt
import numpy as np
import torch
from sympy import log, nsolve, sympify, symbols, solve, Float
from sympy.abc import x
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def shard_num_generate(n, base):
    """
    n:生成多少个用户的shard
    base:中的数据量是多少
    """
    dist = np.random.normal(0, 1, n)  # 使用正态分布
    mn = min(dist)
    dist = [d - mn * (1 + n / 100) for d in dist]  # 加上最大值进行偏移， (n/100)是为了保证最小值不会太小
    s = sum(dist)
    dist = [d * base / s for d in dist]
    dist = [int(round(d)) for d in dist] # 每个用户的shard数量进行取整
    dist[-1] += base - sum(dist)  # 把多余的样本放到最后一个shard里面去
    rand_lst = list(random.sample(range(base), base))
    shard_lst = []
    for d in dist:
        shard_lst.append([])
        for i in range(d):
            if len(rand_lst)==0:
                break
            shard_lst[-1].append(rand_lst.pop())
    print("每个用户的shard数量： ", dist)
    # print("shard的分布： ", shard_lst)
    return shard_lst


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def mySolve(tcp,s,ph,t):
    """
    通过python的
    """
    x = symbols('x')
    # x = sympify(x)
    equation = tcp + (s / (x * log(1 + ph / x))) - t
    solution = solve([equation], [x])
    return solution

def mySolve2(s,h,t,pre_solution,noise,pkd):
    """
    s: 传输的总数据量，对于FL而言，就是模型的总参数量
    h：信道增益
    pkd：传输能量
    pre_solution：上一次计算的结果，本次计算失败，使用上一次的作为结果
    noise:噪声功率
    t: 上传时延

    通过python的
    # 不能直接计算带宽占比，带宽占比值太小，出现无法解出的情况
    upload_rate = b_k * log_2 (1+ (p_k * h_k)/ (noise * b_k)
    upload_delay = S / upload_rate
    =>

    """
    # x = symbols('x')
    # x = sympify(x)
    # equation = (s / (x * log(1 + 0.1*h/(x*(10**(-134/10))),2))) - t
    equation = (s / (x * log(1 + pkd*h/(x*noise),2))) - t
    if equation.has(0):
        print(" 出现 0 ",end="  ")
        return pre_solution
    try:
        solution = nsolve(equation,x,10,verify=False)
        # print((s / (solution * math.log(1 + pkd*h/(solution*noise),2))) - t)
    except Exception as e:
        # print(" 求解异常 ",end=" ")
        # print(e)
        return pre_solution
    if not (isinstance(solution,Float) or isinstance(solution,float)):
        return pre_solution
    # print(t , (s / (solution * math.log(1 + 0.1*h/(solution*(10**(-134/10))),2))) ,sep = '    ')
    return solution


# if __name__ == '__main__':
#     s = 10.
#     p = 3.
#     h = 3.
#     sigma = 20.
#     tcp = 13.
#     b = 10.
#     ph = p*h/sigma
#     t = tcp+(s/(b*log(1+p*h/sigma/b)))
#     # print(t)
#     # print(mySolve(tcp,s,ph,t))
#     tcp = 0.00659624
#     s = 62006.0
#     t = 0.14732768764255275
#     ph = 5.209416304465976e-07
#     print(mySolve2(tcp,s,ph,t))
#     # tcp = 13.
#     # s = 10
#     # t = 35.7
#     # ph = 0.45
#     print(mySolve2(tcp,s,ph,t))
#     print(mySolve(tcp,s,ph,t))
#     # noise = 10**(-134/10)
#     # ttt =np.array(p*h/sigma)
#     # ttt.shape = 1
#     # print(ttt)
#     # tau = (noise * s * math.log(2, math.e)) / (
#     #             (t - tcp) * ((np.array(p*h/sigma) * b) ** 2 / 0.1))
#     # print(s * math.log(2, math.e) /
#     #                                  ((t - tcp) * (tau + lambertw(-tau * math.e ** (-tau)))))
#

# def myPlot(tcp,s,ph,t):
#     end = 10**3
#     xlst = [i/end for i in range(0,end*10)]
#     xlst = xlst[1:]
#     y = [func(x,(tcp,s,ph,t)) for x in xlst]
#     plt.plot(xlst,y)
#     plt.show()


def func(x,args):
    tcp, s, h, t = args
    return tcp + (s / (x * log(1 + 0.1*h/(x*10**(-134/10))))) - t


# 使用二分法寻找函数的零点
def find_root_bisect(func, a, b,args, epsilon=1e-10,):
    # 确保函数在区间[a, b]是单调递增的
    if func(a,args) >= func(b,args):
        raise ValueError("Function is not monotonic")

    # 使用二分法寻找解
    while b - a > epsilon:
        c = a + (b - a) / 2
        if func(c) == 0:
            return c
        elif func(c,args) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2  # 返回近似解


def mySolve3(args):
    (tcp, s, ph, t) = args
    return find_root_bisect(func, 10000000000000, 1e-10,args)


def cal_communication_capability(bandwidth,signal,p,noise):
    return bandwidth*math.log(1+p*signal/(bandwidth*noise))


if __name__ == '__main__':
    # tcp = 0.00659624
    # s = 62006.0
    # t = 0.14732768764255275
    # ph = 5.209416304465976e-07
    # # args = (tcp,s,ph,t)
    # myPlot(tcp,s,ph,t)
    # print(mySolve2(tcp,s,ph,t))
    # print(mySolve3((tcp, s, ph, t)))
    # # print(func(1000,args))
    # x = 16352219829857.3
    # # tcp + (s / (x * math.log(1 + ph / x,2))) - t
    # print(math.log(1+ph/x,2))
    # print(s/(x*math.log(1+ph,2)*10 * 10**6))
    # print(tcp + (s / (x * math.log(1 + ph,2))) - t)
    try:
        1/0
    except Exception as e :
        print(e)
