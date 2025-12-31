#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import math
import random

import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
# import torch
# from sympy import log, nsolve, sympify, symbols, solve, Float
# from sympy.abc import x
# from torchvision import datasets, transforms
# from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
# from sampling import cifar_iid, cifar_noniid
# from src import common
# from src.options import args_parser


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

def shard_num_generate(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max() + 1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1] * len(k_idcs)).
                                                  astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    # 这里必须打乱，否则验证集的标签和训练集的差别太大
    client_idcs = [i.tolist() for i in client_idcs]

    for i in range(len(client_idcs)):
        random.shuffle(client_idcs[i])

    return client_idcs

# def shard_num_generate(n, base):
#     """
#     n:生成多少个用户的shard
#     base:中的数据量是多少
#     """
#     dist = np.random.normal(0, 1, n)  # 使用正态分布
#     mn = min(dist)
#     dist = [d - mn * (1 + n / 100) for d in dist]  # 加上最大值进行偏移， (n/100)是为了保证最小值不会太小
#     s = sum(dist)
#     dist = [d * base / s for d in dist]
#     dist = [int(round(d)) for d in dist] # 每个用户的shard数量进行取整
#     dist[-1] += base - sum(dist)  # 把多余的样本放到最后一个shard里面去
#     rand_lst = list(random.sample(range(base), base))
#     shard_lst = []
#     for d in dist:
#         shard_lst.append([])
#         for i in range(d):
#             if len(rand_lst)==0:
#                 break
#             shard_lst[-1].append(rand_lst.pop())
#     print("每个用户的shard数量： ", dist)
#     # print("shard的分布： ", shard_lst)
#     return shard_lst


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


def plot():
    import matplotlib.pyplot as plt
    import numpy as np  # 用于生成坐标
    fontSize = 22
    # fontSize = 32
    plt.rcParams.update({
        'axes.labelsize': fontSize,  # X/Y轴标签字体大小
        'axes.titlesize': fontSize,  # 标题字体大小
        'legend.fontsize': fontSize,  # 图例字体大小
        'xtick.labelsize': 15,  # X轴刻度标签字体
        'ytick.labelsize': fontSize  # Y轴刻度标签字体
    })

    plt.rcParams['font.sans-serif'] = ['SimSun']  # SimHei 是黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['figure.figsize'] = (8, 4)

    # 读取数据（注意：这里修正了你的数据处理逻辑）
    with open('HSFLWithAlgo2.txt', 'r') as f:
        data1 = eval(f.read())
    with open("HSFLWithCmp2.txt", 'r') as f:
        data2 = eval(f.read())

    data1 = [i*0.0001 for i in data1]
    data2 = [i*0.0001*1.5 for i in data2]

    data11= data1[:]
    data22= data2[:]

    # 数据归一化（修正了data2的处理）
    mx1 = max([i if i!=0 else max(data1) for i in data1])
    data1 = [mx1 - i if i!=0 else 0 for i in data1]
    mx2 = max([i if i!=0 else max(data2) for i in data2 ])  # 原代码这里是 min(data1)，疑似笔误
    data2 = [mx2 - i if i !=0 else 0 for i in data2]

    # 创建画布
    plt.figure(figsize=(10, 6))

    # 生成坐标和条形宽度
    x = np.arange(len(data1))  # 假设data1和data2长度相同
    width = 0.4  # 条形的宽度

    # 绘制并排条形图
    plt.bar(x - width / 2, data1, width=width, label='有带宽优化的等待时延', alpha=0.7)
    plt.bar(x + width / 2, data2, width=width, label='无带宽优化的等待时延', alpha=0.7)
    # 假设 data1 和 data2 是列表或数组
    data1 = np.array(data11)
    data2 = np.array(data22)

    # 获取 data1 非零值的索引和数值
    idx1 = np.where(data1 != 0)[0]
    values1 = data1[idx1]

    # 获取 data2 非零值的索引和数值
    idx2 = np.where(data2 != 0)[0]
    values2 = data2[idx2]

    # 绘制点状图（不同标记区分两组数据）
    plt.scatter(idx1, values1, marker='o', label='有带宽优化的设备时延')  # 圆圈标记
    plt.scatter(idx2, values2, marker='x', label='无带宽优化的设备时延')  # 叉号标记

    # 添加标签和标题
    plt.xlabel('设备索引值')
    plt.ylabel('时延[秒]')
    # plt.title('Comparison between HSFLWithAlgo and HSFLWithCmp')
    plt.xticks(x)  # 显示所有数据点的刻度
    plt.legend(loc='upper left',
               # bbox_to_anchor=(0.4, 1.0),ncol=2
               framealpha=0.3  # 设置透明度（0.0~1.0）
               )

    # 显示图表
    plt.tight_layout()

    img_path = r"C:\Users\lxf_98\data\OneDrive\文档\硕士\报告\毕业论文\图片"
    file_path="c3_bandAllocation"
    file_path = img_path + "\\" + file_path
    file_path = file_path + {"CN": "cn", "EN": "en"}["CN"]
    # plt.tight_layout(rect=[0, 0, 1, 0.85])  # rect参数控制有效区域
    # 保存为 PNG 文件
    res = plt.savefig(file_path + ".png", bbox_inches='tight', pad_inches=0)

    # 保存为 PDF 文件
    res = plt.savefig(file_path + ".pdf", bbox_inches='tight', pad_inches=0)

    plt.show()

# # 调用函数生成图表
# plot()
def plot2():
    import matplotlib.pyplot as plt
    import numpy as np
    fontSize = 22
    plt.rcParams.update({
        'axes.labelsize': fontSize,
        'axes.titlesize': fontSize,
        'legend.fontsize': fontSize,
        'xtick.labelsize': 15,
        'ytick.labelsize': fontSize,
        'font.sans-serif': ['SimSun'],
        'axes.unicode_minus': False,
        'figure.figsize': (10, 6)  # 保持画布大小
    })

    # 读取数据
    with open('HSFLWithAlgo2.txt', 'r') as f:
        data1 = eval(f.read())
    with open("HSFLWithCmp2.txt", 'r') as f:
        data2 = eval(f.read())

    data1 = [i * 0.0001 for i in data1]
    data2 = [i * 0.0001 * 1.5 for i in data2]
    data11, data22 = data1.copy(), data2.copy()

    # 数据归一化
    mx1 = max([i if i != 0 else max(data1) for i in data1])
    data1 = [mx1 - i if i != 0 else 0 for i in data1]
    mx2 = max([i if i != 0 else max(data2) for i in data2])
    data2 = [mx2 - i if i != 0 else 0 for i in data2]

    # 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制条形图
    x = np.arange(len(data1))
    width = 0.4
    ax.bar(x - width / 2, data1, width=width, label='有带宽优化的等待时延', alpha=0.7)
    ax.bar(x + width / 2, data2, width=width, label='无带宽优化的等待时延', alpha=0.7)

    # 绘制散点图
    data1 = np.array(data11)
    data2 = np.array(data22)
    idx1 = np.where(data1 != 0)[0]
    idx2 = np.where(data2 != 0)[0]
    ax.scatter(idx1, data1[idx1], marker='o', label='有带宽优化的设备时延')
    ax.scatter(idx2, data2[idx2], marker='x', label='无带宽优化的设备时延')

    # 设置坐标轴标签
    ax.set_xlabel('设备索引值')
    ax.set_ylabel('时延[秒]')
    ax.set_xticks(x)

    # 关键调整1：压缩主图内容（下移图形）
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 0.8)  # 将Y轴上限压缩到80%，为图例腾出空间

    # 关键调整2：将图例放置在axes内部
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),  # 定位到axes顶部中央
        ncol=2,
        frameon=True
    )

    # 关键调整3：通过subplots_adjust控制边距（不改变axes大小）
    plt.subplots_adjust(top=0.75)  # 顶部边距留出25%空间

    # 保存文件
    img_path = r"C:\Users\lxf_98\data\OneDrive\文档\硕士\报告\毕业论文\图片"
    file_path = img_path + "\\c3_bandAllocation" + {"CN": "cn", "EN": "en"}["CN"]
    plt.savefig(file_path + ".png", bbox_inches='tight', pad_inches=0)
    plt.savefig(file_path + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.show()



import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def colorBar():
    # 模拟数据
    before = {}
    floor = {}
    after = {}



    # 假设从CSV文件中读取数据
    with open("aaaround.csv", 'r', encoding="utf-8") as f:
        data = f.read()
        cp = data.strip().split("\n")
        for c in cp:
            d = c.strip().split(",")
            lst = [float(d[2]),float(d[3]),float(d[4])]
            lst.sort()
            before[str(d[0]) + "," + str(d[1])] = lst[2]
            floor[str(d[0]) + "," + str(d[1])] = lst[1]
            after[str(d[0]) + "," + str(d[1])] = lst[0]

    # 提取数据用于绘图
    x_unique = sorted(list(set(float(key.split(',')[0]) for key in before.keys())))
    y_unique = sorted(list(set(float(key.split(',')[1]) for key in before.keys())))

    # 创建网格
    X, Y = np.meshgrid(x_unique, y_unique)

    # 初始化Z值
    Z_before = np.zeros_like(X)
    Z_floor = np.zeros_like(X)
    Z_after = np.zeros_like(X)

    data = {}

    # 填充Z值
    for i, x in enumerate(x_unique):
        for j, y in enumerate(y_unique):
            key = f"{int(x)},{int(y)}"
            if key in before:
                Z_before[j, i] = before[key]
                before_val = before[key]
            if key in floor:
                Z_floor[j, i] = floor[key]
                floor_val = floor[key]
            if key in after:
                Z_after[j, i] = after[key]
                after_val = after[key]
            (before_val,after_val,floor_val) = sorted((before_val,after_val,floor_val))
            data[f"after_rho1_{int(y)},rho2_{int(x)}"] = after_val
            data[f"floor_rho1_{int(y)},rho2_{int(x)}"] = floor_val
            data[f"before_rho1_{int(y)},rho2_{int(x)}"] = before_val
            print("before",before_val,"after",after_val,"floor",floor_val)
    sio.savemat("matlabData/matlab/batchRoundAlgo.mat", data)

    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    # Before 数据集，使用圆形标记
    ax.scatter(X, Y, Z_before, c='blue', marker='o', label='Before', alpha=0.7)
    # Floor 数据集，使用方形标记
    ax.scatter(X, Y, Z_floor, c='green', marker='s', label='Floor', alpha=0.7)
    # After 数据集，使用三角形标记
    ax.scatter(X, Y, Z_after, c='red', marker='^', label='After', alpha=0.7)

    # 添加图例和标签
    ax.set_xlabel('rho1')
    ax.set_ylabel('rho2')
    ax.set_zlabel('Value')
    ax.set_title('3D Scatter Plot of Before, Floor, and After')

    # 显示图形
    plt.legend()
    plt.show()

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
    # try:
    #     1/0
    # except Exception as e :
    #     print(e)
    # plot()
    # plot2()

    #
    # rhoCp=[
    #     (3,500),
    #     (4,500),
    #     (5,500),
    #     (6,500),
    #     (3,2000),
    #     (4,2000),
    #     (5,2000),
    #     (6,2000),
    #     (3,5000),
    #     (4,5000),
    #     (5,5000),
    #     (6,5000),
    # ]
    # with open("aaaround.csv",'r') as f:
    #     data = f.read()
    #     cp = data.strip().split("\n")
    #     before = {}
    #     floor = {}
    #     after = {}
    #     for c in cp:
    #         d = c.strip().split(",")
    #         before[str(d[0])+","+str(d[1])] = d[2]
    #         floor[str(d[0])+","+str(d[1])] = d[3]
    #         after[str(d[0])+","+str(d[1])] = d[4]

    colorBar()