import math
import random
from typing import List

import numpy as np

from seed_everything import seed_everything


def compute_capacity_rand_generate(n: int) -> List[int]:
    """
    传入当前的客户端数量，随机生成每个客户端的计算能力，生成后固定不动
    使用[1,1.6]*10^9 cycles/s 的区间进行均匀分布生成，间隔100*10^6
    返回随机生成的结果
    """
    base = 10 ** 8
    # kappa = 16 # 单位是cycles/FLOPs ，即计算一个FLOPs需要多少个计算周期
    kappa = 16 # 单位是cycles/FLOPs ，即计算一个FLOPs需要多少个计算周期
    dist = np.random.uniform(1, 9, n)
    # 均匀分布生成会有小数存在，但是在[1,2) ， [2,3) 会有相同的概率，那么一直到[8,9)都是相同的概率，使用floor之后取得每个相同概率slot的下限值
    dist = [math.floor(d) * base /kappa for d in dist]
    idx = random.randint(0,len(dist)-1)
    div = 1
    # print("*"*20+str(div)+"*"*20)
    dist[idx]/=div
    dist.append(50*base/kappa)  # 服务器的计算性能是普通客户端的五十倍
    return dist


def pos_rand_generate(n: int) -> List[complex]:
    """
    传入当前的客户端数量，随机生成每个客户端的坐标，假定中心服务器都是在(0,0)位置，每一个round生成一次
    使用[30，100] m 的区间进行均匀分布生成
    返回随机生成的结果
    """
    x_lst = np.random.uniform(2,11,n)
    x_lst = [math.floor(x) for x in x_lst]
    y_lst = np.random.uniform(2,11,n)
    y_lst = [math.floor(y) for y in y_lst]
    # 均匀分布生成会有小数存在，但是在[1,2) ， [2,3) 会有相同的概率，那么一直到[8,9)都是相同的概率，使用floor之后取得每个相同概率slot的下限值

    return [x-1j*y for x,y in zip(x_lst,y_lst)]


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
    dist = [round(d) for d in dist] # 每个用户的shard数量进行取整
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

def state_generate(self,init=False,action = None,accuracy = 0,step_num=0,loss=None,d=None,cnt = None):
    # alternative state : len , speed, loss, action , acc ,step
    u_min = min(self.uplink_speed)
    u_max = max(self.uplink_speed)-u_min
    l_min = min(self.sample_len)
    l_max = max(self.sample_len)-l_min
    state_list  = []
    if init:
        dic = {"len": [0]*self.client_total,
               "speed": [0]*self.client_total,
               "loss": [0]*self.client_total,
               'action': [0]*self.client_total,
               "acc": [0],
               'd':[0]*self.client_total,
               "step": [0],
               'cnt':[0]*self.client_total}

    else:
        a_min = min(action)
        a_max = max(action)-a_min
        dic = {"len": [(l-l_min)/(1 if l_max==0 else l_max ) for l in self.sample_len],
               "speed": [(i-u_min)*2/u_max-1 for i in self.uplink_speed],
               # "loss": [(l-loss_min)*2/loss_max-1 for l in loss],
               # TODO loss 如何处理是个问题
               "loss": loss,
               # 'action': [(a-a_min)*2/a_max-1 for a in action],
               'action': action,
               "acc": [accuracy],
               "cnt": cnt ,
               'd':np.log10(np.array([np.random.exponential(1e-4 /pow(s,4)) for s in d])).tolist(),
               "step": [step_num+1]}
    for state in self.config.state_composition:
        state_list.extend(dic[state])

    # Todo
    print("state:",state_list)
    return state_list





if __name__ == '__main__':
    # seed_everything(2048)
    # print(shard_num_generate(100, 1000))
    print(128.1 + 37.6*math.log((10**-3)*100,10))