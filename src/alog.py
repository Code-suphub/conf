import math
from bisect import bisect, bisect_left

import numpy as np
from sympy import Float

from utils import mySolve2
from scipy.optimize import newton

# 导入日志模块
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
# 设置日志输出格式
formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)


def equation(b_k,p_k,h_k,noise,S):
    upload_rate = b_k*math.log2(1+(p_k*h_k)/noise*b_k)
    upload_delay = S/upload_rate
    return upload_delay


def newton_solve(p_k,h_k,noise,S,upload_delay,bandwidth):
    # 定义方程
    def equation_to_solve(b_k):
        # b_k = max(b_k, 1e-6)
        b_k *= bandwidth
        term = (p_k * h_k) / (noise * b_k)
        if term<=-1:
            print(p_k,h_k,noise,b_k)
            print(term)
            print(p_k*h_k)
            print(noise*b_k)
        upload_rate = b_k * math.log2(1 + term)
        return upload_rate - S / upload_delay

    # # 自定义牛顿法实现，确保 b_k 始终为正值
    # def custom_newton(func, x0, fprime=None, tol=1.48e-08, maxiter=50):
    #     for _ in range(maxiter):
    #         fval = func(x0)
    #         fder = fprime(x0) if fprime is not None else (func(x0 + tol) - fval) / tol
    #         if fder == 0:
    #             raise ValueError("Derivative was zero.")
    #         x1 = x0 - fval / fder
    #         x1 = max(x1, 1e-6)  # 确保新的猜测值是正的
    #         if abs(x1 - x0) < tol:
    #             return x1
    #         x0 = x1
    #     raise RuntimeError("Failed to converge after {} iterations.".format(maxiter))

    initial_guess = max(1e-10, S / (upload_delay * math.log2(1 + (p_k * h_k) / noise))/ bandwidth)
    b_k_solution = newton(equation_to_solve,initial_guess)
    # print(b_k_solution)
    return b_k_solution


class Algo:
    def __init__(self, fl_lst, sl_lst, capacity, bandwidth, signal_cap, model_param, activations, user_groups, args,
                 flops, compute_list, sample_size, pku,rho2):
        # 下面是一次实验的设定，每一次实验不发生变化
        self.pkd = 1
        self.noise = 10 ** (-134 / 10)
        self.flops = flops
        self.args = args
        self.Bandwidth = bandwidth
        self.model_param = model_param
        self.activations = activations
        self.sample_size = sample_size
        self.pku = pku
        self.download_bandwidth = 1.4 * 10 ** 6
        self.rho2 = 1/rho2

        self.user_groups = user_groups  # 是否每一个round发生变化待定

        # 下面是计算和通信（链路情况）每一个round发生变化
        self.compute_list = compute_list
        self.capacity = capacity
        self.signal_cap = [s[0] for s in signal_cap]

        # 上面是不变参量，下面是每一个round多次发生变化的参量
        self.fl_lst = fl_lst
        self.sl_lst = sl_lst
        self.cutlayer_lst = [0 for _ in range(len(fl_lst))]
        self.b0 = 1 / len(self.sl_lst) * sum(self.sl_lst)
        self.fl_band_ratio = [1 / sum(self.fl_lst) if self.fl_lst[i] != 0 else 0 for i in range(len(self.fl_lst))]
        self.dic = {"type": set()}
        self.batch_size_lst = [1]*len(self.user_groups)
        self.lambda_lst = [1] *len(self.fl_lst)
        self.mu = 1
        self.user_num = len(self.user_groups)
        # self.b_k_values = np.linspace(20, 200000, 40000)[::-1]

        # # 下面是生成当前round 的初始时延上届，以纯 SL 的时延作为上届值
        # self.slr = [1 / args.num_users for _ in range(self.args.num_users)]
        # # 每个sl用户均分带宽，这个slr是不每一轮使用一次的，因为不会出现每个用户都是sl
        # self.sluc = [capacity[i] * self.slr[i] for i in range(args.num_users)]  # 算出均匀分配带宽下的通信容量（因为这里的capacity
        # # 都是全部容量下计算得到的结果值）
        # self.sldc = [
        #     (self.Bandwidth * np.log2(1 + self.pkd * signal_cap[i] / Bandwidth / self.noise) * self.slr[i]).item() for i
        #     in
        #     range(args.num_users)]  # 计算 sl 下行链路的通信容量
        # self.sltd = [model_param[4] / uc + model_param[4] / dc
        #              + activations[4][-1] * len(user_groups[i]) / uc + activations[4][-1] * len(user_groups[i]) / dc
        #              for i, (uc, dc) in enumerate(zip(self.sluc, self.sldc))]  # 每个客户端的通信时延,sl还有激活值的时延
        # self.sljd = [self.flops[-1] / self.compute_list[i] for i in range(len(sl_lst))]
        # self.sld = [t + j for t, j in zip(self.sltd, self.sljd)]  #
        # self.sldUp = sum(self.sld)  # 这里是计算上限时延，即所有用户都是SL用户，并且都是所有数据都在本地计算

        # 下面是初始每个客户端的初始状态计算能力，通信能力和计算时延，通信时延
        self.fld, self.sld = self.cal_delay()

    # def update_device_partition

    def binary_b0(self,cut_,band_,batch_lst=None,cutlay_lst = None):
        if batch_lst==None:
            batch_lst = [len(i) for i in self.user_groups]

        # 二分搜索 b_0
        bu, bd = 1, 0
        self.sl_bandwidth, self.fl_bandwidth = self.b0 * self.Bandwidth, (1 - self.b0) * self.Bandwidth
        while True:
            """
            二分搜索 b0 ，没问题 
            b0:     二分搜索的 sl 客户带宽分配的结果值
            bu:     二分搜索的 sl 客户带宽分配的上界值，初始为全部带宽1
            bl:     二分搜索的 sl 客户带宽分配的上界值，初始为0
            flc:    fl 用户的通信容量（在找b0时假定所有带宽均匀分配）
            slc:    sl 用户的通信容量（sl公用一个时延）
            fltd:   fl用户的传输时延
            sltd:   sl用户的传输时延
            sljd:   sl用户的计算时延
            fljd:   fl用户的计算时延（因为sl用户都是本地计算，计算时延与本round的任何调度无关，所以在开头计算）
            fld:    fl用户的总时延（最大值）
            sld:    sl用户的总时延（累加值）
            """
            self.b0 = (bu + bd) / 2
            self.sl_bandwidth, self.fl_bandwidth = self.b0 * self.Bandwidth, (1 - self.b0) * self.Bandwidth
            fld, sld = self.cal_delay()
            if cut_:
                # self.cut_decision(batch_lst)
                self.cutlayer_lst = cutlay_lst if cutlay_lst!=None else  [0 if self.sl_lst[i]==1 else [-1000,-1000]  for i in range(self.user_num) ]
            if band_:
                self.bandwidth_with_delay_bin()
            fld, sld = self.cal_delay()

            if abs(fld - sld) < 0.003 or bu==bd:
                break
            elif fld > sld:
                bu = self.b0
            else:
                bd = self.b0
        return self.b0

    def cut_decision(self,batch_lst):
        """
        进行分割层的优化
        """
        self.cutlayer_lst = []
        sld = 0
        for i, a in enumerate(self.sl_lst):  # 通过迭代每一个分割层作为分割点，找到每一个sl用户的最优分割层
            """
            a:      每个客户端的行为，如果在sl_lst中表现为1，则表示是SL用户
            cres:   在当前分割层cc所需要的时延，用来找到最小时延
            cc:     分割层的结果值，默认全部由服务器训练，即-1（因为索引关系，activations中索引从0~L-1
            c:      可选的分割层
            cut_layLst: 将分割层和当前分割成下
            """
            if a == 1:
                default_delay = batch_lst[i] * self.sample_size / self.sl_uplink_cap[i] + \
                                self.flops[-1] / self.compute_list[-1]  # 默认直接由服务器训练
                cc = len(self.activations) - 1
                for c in range(len(self.activations)):
                    new_delay = (batch_lst[i] * self.activations[c][-1] + self.model_param[c]) / self.sl_uplink_cap[i] \
                                + (batch_lst[i] * self.activations[c][-1] + self.model_param[c]) / self.sl_download_cap[i] \
                                + self.flops[c] / self.compute_list[i] + (self.flops[-1] - self.flops[c]) / self.compute_list[-1]
                    if new_delay < default_delay:
                        default_delay = new_delay
                        cc = c
                self.cutlayer_lst.append(cc)
                sld += default_delay
            else:
                self.cutlayer_lst.append([-1000, -1000])
        return sld

    def bandwidth_with_delay_bin(self):
        """
        通过二分搜索时延的上下界找到每个设备的带宽分配
        """
        if sum(self.fl_lst) == 0:
            return
        # 计算当前的时延上界，为平均分配的时候的带宽，因为下行使用广播信道，所以下行时延不在优化范围内
        fl_compute_delay, _ = self.cal_compute_delay()
        fl_uplink_trans_delay, fl_download_trans_delay, _, _ = self.cal_trans_delay()
        fl_non_uplink_delay = [dt + j for dt, j in zip(fl_download_trans_delay, fl_compute_delay)]

        fld = [(ut + j) for ut, j in zip(fl_non_uplink_delay, fl_uplink_trans_delay)]
        # fl 所有用户的时延上界就是
        t_upper_bound = max(fld)
        t_lower_bound = max(fl_non_uplink_delay)
        self.bandwidth_lst = [band * (1 - self.b0) * self.Bandwidth for band in self.fl_band_ratio]
        cnt = 0
        # equation_values = [
        #     [equation(b_k, self.signal_cap[i], self.pku,self.noise, self.model_param[-1]) for b_k in self.b_k_values] if self.fl_lst[i]==1 else 0 for i in range(self.user_num)
        # ]
        while True:
            """
            t_upper_bound:     fl时延的上界，初始值均匀分配带宽时的fl时延最大值
            t_lower_bound:     fl时延下界，初始值为fl计算时延的最大值（如果没有通信时延，即约等于无线带宽）
            tm:     fl时延的中值

            flc:    fl带宽，初始值为均匀分配
            fltd:   fl的通信时延
            fld:    fl的时延
            """
            fl_band_ratio = []
            tm = (t_upper_bound + t_lower_bound) / 2
            for i, a in enumerate(self.fl_lst):
                if a == 1:
                    # # fl 用户，d =
                    # (s, p, t) = self.model_param[-1], self.signal_cap[i], tm - fl_non_uplink_delay[i]
                    upload_delay = tm - fl_non_uplink_delay[i]
                    newton_b_k_ratio = newton_solve(self.signal_cap[i],self.pku,self.noise,self.model_param[-1],upload_delay ,self.Bandwidth*(1-self.b0))

                    # print("newton_b_k",newton_b_k_ratio* self.Bandwidth*(1-self.b0))
                    # def equation_to_solve(b_k):
                    #     # b_k = max(b_k, 1e-6)
                    #     term = (self.signal_cap[i] * self.pku) / (self.noise * b_k)
                    #     upload_rate = b_k * math.log2(1 + term)
                    #     print("target delay: ",tm," actual delay: ",self.model_param[-1]/upload_rate +fl_non_uplink_delay[i]," gap: ", tm -(self.model_param[-1]/upload_rate +fl_non_uplink_delay[i]) )
                    #     # print("upload_delay:  ",upload_delay,"  actual_delay:  ",self.model_param[-1]/upload_rate,"  gap:  ",upload_delay - self.model_param[-1]/upload_rate)
                    # equation_to_solve(newton_b_k_ratio* self.Bandwidth*(1-self.b0))
                    fl_band_ratio.append(newton_b_k_ratio)
                else:
                    # sl用户
                    fl_band_ratio.append(0)
            try:
                self.fl_band_ratio_sum = float(sum(fl_band_ratio))
            except Exception as e:
                print(e)
                print(isinstance(fl_band_ratio[0], Float))
                print(isinstance(fl_band_ratio[0], float))
                # print(isinstance(bandwidth_lst[0],float))
                input("错误啦")
            try:
                # if abs(bandwidth_sum - self.Bandwidth*(1-self.b0))<10000:
                #     break
                if self.fl_band_ratio_sum > 1:  # 大于预算带宽，说明给定每个用户的时间太短，需要的带宽过高，提升时间下界
                    t_lower_bound = tm
                elif self.fl_band_ratio_sum < 1:  # 小于预算带宽
                    t_upper_bound = tm
                else:
                    cnt += 1
            except Exception as e:
                print(e)
                input(" 发生了错误 ")
            # else:
            #     break
            gap = t_upper_bound - t_lower_bound
            if gap < 0.00001:
                break
            self.fl_band_ratio = fl_band_ratio
            fl_uplink_trans_delay, fl_download_trans_delay, _, _ = self.cal_trans_delay()  # 每次带宽分配更新之后，fl的下行时延会发生变化，这里需要进行更新
            fl_non_uplink_delay = [dt + j for dt, j in zip(fl_download_trans_delay, fl_compute_delay)]
            fld = [dt + j for dt, j in zip(fl_non_uplink_delay, fl_uplink_trans_delay)]
            # t_lower_bound = min(t_lower_bound, max(fl_non_uplink_delay))
            a = 1
        # if bandwidth_sum>self.Bandwidth * (1-self.b0):  # 可能超预算
        self.fl_band_ratio = fl_band_ratio
        _, fl_download_trans_delay, _, _ = self.cal_trans_delay()  # 每次带宽分配更新之后，fl的下行时延会发生变化，这里需要进行更新
        fl_non_uplink_delay = [dt + j for dt, j in zip(fl_download_trans_delay, fl_compute_delay)]
        fld = [dt + j for dt, j in zip(fl_non_uplink_delay, fl_uplink_trans_delay)]
        a= 1

    def bandwidth_with_delay_bin_with_GPT(self):
        """
        通过二分搜索时延的上下界找到每个设备的带宽分配
        """
        if sum(self.fl_lst) == 0:
            return
        # 计算当前的时延上界，为平均分配的时候的带宽，因为下行使用广播信道，所以下行时延不在优化范围内
        fl_compute_delay, _ = self.cal_compute_delay()
        fl_uplink_trans_delay, fl_download_trans_delay, _, _ = self.cal_trans_delay()
        fl_non_uplink_delay = [dt + j for dt, j in zip(fl_download_trans_delay, fl_compute_delay)]

        fld = [(ut + j) for ut, j in zip(fl_non_uplink_delay, fl_uplink_trans_delay)]
        # fl 所有用户的时延上界就是
        t_upper_bound = max(fld)
        t_lower_bound = max(fl_non_uplink_delay)
        self.bandwidth_lst = [band * (1 - self.b0) * self.Bandwidth for band in self.fl_band_ratio]
        cnt = 0
        while True:
            """
            t_upper_bound:     fl时延的上界，初始值均匀分配带宽时的fl时延最大值
            t_lower_bound:     fl时延下界，初始值为fl计算时延的最大值（如果没有通信时延，即约等于无线带宽）
            tm:     fl时延的中值

            flc:    fl带宽，初始值为均匀分配
            fltd:   fl的通信时延
            fld:    fl的时延
            """
            target_delay = (t_upper_bound + t_lower_bound) / 2
            bandwidth_lst = []
            for i, a in enumerate(self.fl_lst):
                if a == 1:
                    # # fl 用户，d =
                    # (s, p, t) = self.model_param[-1], self.signal_cap[i], tm - fl_non_uplink_delay[i]
                    newton_b_k = newton_solve(self.signal_cap[i],self.pku,self.noise,self.model_param[-1],target_delay)
                    # print("newton_b_k",newton_b_k)
                    # (s, p, t) = self.model_param[-1], self.signal_cap[i], tm - fl_non_uplink_delay[i]
                    # mysolve2 = mySolve2(s, p, t, self.bandwidth_lst[i], self.noise, self.pku)
                    # print("mysolve2",mysolve2)
                    bandwidth_lst.append(newton_b_k)
                else:
                    # sl用户
                    bandwidth_lst.append(0)
            self.bandwidth_sum = float(sum(bandwidth_lst))
            self.bandwidth_lst = bandwidth_lst
            self.fl_band_ratio = [band / self.bandwidth_sum for band in bandwidth_lst]

            # 计算每个设备的时延
            _, fl_download_trans_delay, _, _ = self.cal_trans_delay()
            fl_non_uplink_delay = [dt + j for dt, j in zip(fl_download_trans_delay, fl_compute_delay)]
            fl__delay = [dt + j for dt, j in zip(fl_non_uplink_delay, fl_uplink_trans_delay)]

            # 检查所有设备的时延是否接近目标时延
            delays_diff = [abs(delay - target_delay) for delay in fl__delay if self.fl_lst[i] == 1]
            if all(diff < 0.001 for diff in delays_diff):  # 设定一个小的容差值
                break

            # 根据时延差异调整目标时延
            if any(delay > target_delay for delay in fl__delay if self.fl_lst[i] == 1):
                t_lower_bound = target_delay
            else:
                t_upper_bound = target_delay

        # if bandwidth_sum>self.Bandwidth * (1-self.b0):  # 可能超预算
        self.bandwidth_lst = [band * (self.fl_bandwidth / self.bandwidth_sum) for band in self.bandwidth_lst]
        self.fl_band_ratio = [band / self.bandwidth_sum for band in bandwidth_lst]
        _, fl_download_trans_delay, _, _ = self.cal_trans_delay()  # 每次带宽分配更新之后，fl的下行时延会发生变化，这里需要进行更新
        fl_non_uplink_delay = [dt + j for dt, j in zip(fl_download_trans_delay, fl_compute_delay)]
        fl__delay = [dt + j for dt, j in zip(fl_non_uplink_delay, fl_uplink_trans_delay)]
        a= 1

    def bandwidth(self):
        """
        FL 用户的带宽分配
        暂时都不进行迭代
        """
        # TODO 如何进行加速

        if sum(self.fl_lst)==0:
            return
        fl_compute_delay, _ = self.cal_compute_delay()
        fl_uplink_trans_delay, fl_download_trans_delay, _, _ = self.cal_trans_delay()
        fl_non_uplink_delay = [dt + j for dt, j in zip(fl_download_trans_delay, fl_compute_delay)]
        fld = [(ut + j) * 10 for ut, j in zip(fl_non_uplink_delay, fl_uplink_trans_delay)]
        # fl 所有用户的时延上界就是
        t_upper_bound = max(fld)
        t_lower_bound = max(fl_non_uplink_delay)
        self.bandwidth_lst = [band * (1 - self.b0) * self.Bandwidth for band in self.fl_band_ratio]
        cnt  =0
        while True:
            """
            t_upper_bound:     fl时延的上界，初始值均匀分配带宽时的fl时延最大值
            t_lower_bound:     fl时延下界，初始值为fl计算时延的最大值（如果没有通信时延，即约等于无线带宽）
            tm:     fl时延的中值
            
            flc:    fl带宽，初始值为均匀分配
            fltd:   fl的通信时延
            fld:    fl的时延
            """
            bandwidth_lst = []
            tm = (t_upper_bound + t_lower_bound) / 2
            for i, a in enumerate(self.fl_lst):
                if a == 1:
                    # fl 用户，
                    (s, p, t) = self.model_param[-1], self.signal_cap[i], tm - fl_non_uplink_delay[i]
                    bandwidth_lst.append(mySolve2(s, p, t, self.bandwidth_lst[i], self.noise, self.pku))
                else:
                    # sl用户
                    bandwidth_lst.append(0)
            try:
                self.bandwidth_sum = float(sum(bandwidth_lst))
            except Exception as e:
                print(e)
                print(isinstance(bandwidth_lst[0],Float))
                print(isinstance(bandwidth_lst[0],float))
                # print(isinstance(bandwidth_lst[0],float))
                input("错误啦")
            bandwidth_sum = self.bandwidth_sum
            self.dic["type"].add(type(self.bandwidth_sum))
            # print(isinstance(bandwidth_sum,Float))
            try:
                # if abs(bandwidth_sum - self.Bandwidth*(1-self.b0))<10000:
                #     break
                if self.bandwidth_sum > self.fl_bandwidth:  # 大于预算带宽，说明给定每个用户的时间太短，需要的带宽过高，提升时间下界
                    t_lower_bound = tm
                elif self.bandwidth_sum < self.fl_bandwidth:  # 小于预算带宽
                    t_upper_bound = tm
                else:
                    cnt+=1
            except Exception as e:
                print(e)
                input(" 发生了错误 ")
            # else:
            #     break
            gap = t_upper_bound - t_lower_bound
            if gap < 0.001 or cnt>20:
                break
            self.bandwidth_lst = bandwidth_lst
            self.fl_band_ratio = [band / self.bandwidth_sum for band in bandwidth_lst]
            fl_band_ratio = self.fl_band_ratio
            _, fl_download_trans_delay, _, _ = self.cal_trans_delay()  # 每次带宽分配更新之后，fl的下行时延会发生变化，这里需要进行更新
            fl_non_uplink_delay = [dt + j for dt, j in zip(fl_download_trans_delay, fl_compute_delay)]
            t_lower_bound = min(t_lower_bound, max(fl_non_uplink_delay))
            a = 1
        # if bandwidth_sum>self.Bandwidth * (1-self.b0):  # 可能超预算
        self.bandwidth_lst = [band * (self.fl_bandwidth / bandwidth_sum) for band in self.bandwidth_lst]
        self.fl_band_ratio = [band / self.bandwidth_sum for band in bandwidth_lst]

    def update_partition(self, fl_lst, sl_lst):
        """
        每次更新用户之后需要进行对象数据的更新，其中对象的不变量不需要更新，如计算能力
        而时变量需要更新，如计算能力，而计算能力是通过每个用户的带宽占比表征
        其中因为sl的带宽占比 b0 是在每次计算通信时延动态生成，所以不需要更新
        而 fl 带宽占比依赖于 fl 用户的数量，需要进行更新
        同时 sl 的分割曾需要更新
        :param fl_lst:
        :param sl_lst:
        :return:
        """
        self.fl_lst = fl_lst
        self.sl_lst = sl_lst
        self.fl_band_ratio = [1 / sum(self.fl_lst) if self.fl_lst[i] != 0 else 0 for i in range(len(self.fl_lst))]
        self.cutlayer_lst = [1 for _ in range(self.args.num_users)]

    def cal_compute_delay(self,batch_lst = None):
        if batch_lst is None:
            batch_lst = [len(i) for i in self.user_groups]
        batch_lst = [i*10 for i in batch_lst]

        # 计算每个设备的计算时延
        # return [self.flops[-1] / self.compute_list[i] if self.fl_lst[i] == 1 else 0 for i in range(len(self.fl_lst))], \
        #        [(self.flops[self.cutlayer_lst[i]] / self.compute_list[i]) +
        #         (self.flops[-1] - self.flops[self.cutlayer_lst[i]]) / self.compute_list[-1]
        #         if self.sl_lst[i] == 1 else 0 for i in range(len(self.sl_lst))]  # 每个客户端的计算时延 , sl 包含本地计算时延和服务器计算时延
        # 对于 fl 来说，是完整模型的计算量,上面的没有计算上当前设备的数据量，每个样本一个计算量
        # 对于 sl 来说，是1~cutlay为设备侧的计算量， cutlayer ~ L 是服务器测的计算量
        return [self.flops[-1] / self.compute_list[i] * batch_lst[i] if self.fl_lst[i] == 1 else 0 for i in range(len(self.fl_lst))], \
            [((self.flops[self.cutlayer_lst[i]] / self.compute_list[i]) +
             (self.flops[-1] - self.flops[self.cutlayer_lst[i]]) / self.compute_list[-1]) * batch_lst[i]
             if self.sl_lst[i] == 1 else 0 for i in range(len(self.sl_lst))]  # 每个客户端的计算时延 , sl 包含本地计算时延和服务器计算时延

    def cal_trans_delay(self,batch_lst = None):
        """
        :return: FL 上行时延、下行时延
                SL 上行时延，下行时延
        """
        if batch_lst is None:
            batch_lst = [len(i) for i in self.user_groups]

        batch_lst = [i*10 for i in batch_lst]

        # FL用户的带宽率是每个用户自己拥有的，sl的带宽率是属于共享的，只是多个副本
        flr, slr = [self.fl_band_ratio[i] * (1 - self.b0) if self.fl_lst[i] == 1 else 0 for i in
                    range(len(self.fl_lst))], \
                   [self.b0 if self.sl_lst[i] == 1 else 0 for i in range(len(self.sl_lst))]  # 在rate中将不在本簇的用户置0

        # 上行速率是每个的带宽计算得到
        fl_uplink_cap, self.sl_uplink_cap = \
            [self.cal_communication_capability(i, self.pku, flr[i] * self.Bandwidth) if self.fl_lst[i] == 1 else 0 for i
             in range(len(self.fl_lst))], \
            [self.cal_communication_capability(i, self.pku, slr[i] * self.Bandwidth) if self.sl_lst[i] == 1 else 0 for i
             in range(len(self.sl_lst))]
        # 下行速率对于FL而言使用广播带宽，对于SL而言使用的是各自的带宽使用
        fl_download_cap, self.sl_download_cap = \
            [self.cal_communication_capability(i, self.pkd, self.download_bandwidth)
             if self.fl_lst[i] == 1 else 0 for i in range(self.args.num_users)], \
            [self.cal_communication_capability(i, self.pkd, slr[i]* self.Bandwidth)
             if self.sl_lst[i] == 1 else 0 for i in range(self.args.num_users)]

        # [(fl_band[i] * np.log2(1 + self.pkd * self.signal_cap[i] / (fl_band[i] * self.noise)))
        #  .item() if self.fl_lst[i] == 1 else 0 for i in range(self.args.num_users)], \
        # [(sl_band[i] * np.log2(1 + self.pkd * self.signal_cap[i] / (sl_band[i] * self.noise)))
        #  .item() if self.sl_lst[i] == 1 else 0 for i in range(self.args.num_users)]

        # fl上行传输时延和下行传输时延都是直接模型参数总量除上对应的上下行速率即可
        # SL 上行传输时延是客户端侧模型+激活值/梯度值（这里的下行的梯度值直接使用激活值替代了）除以对应的上下行速率
        fl_uplink_trans_delay, fl_download_trans_delay, sl_uplink_trans_delay, sl_download_trans_delay = \
            [self.model_param[-1] / uc if self.fl_lst[i] != 0 else 0 for i, (uc, dc) in
             enumerate(zip(fl_uplink_cap, fl_download_cap))], \
            [self.model_param[-1] / dc if self.fl_lst[i] != 0 else 0 for i, (uc, dc) in
             enumerate(zip(fl_uplink_cap, fl_download_cap))], \
            [(self.model_param[self.cutlayer_lst[i]] +
              self.activations[self.cutlayer_lst[i]][-1] * len(batch_lst)) / uc
             if self.sl_lst[i] != 0 else 0 for i, (uc, dc) in
             enumerate(zip(self.sl_uplink_cap, self.sl_download_cap))], \
            [(self.model_param[self.cutlayer_lst[i]] +
              self.activations[self.cutlayer_lst[i]][-1] * len(batch_lst)) / dc
             if self.sl_lst[i] != 0 else 0 for i, (uc, dc) in
             enumerate(zip(self.sl_uplink_cap, self.sl_download_cap))]  # 每个客户端的通信时延,sl还有激活值的时延

        return fl_uplink_trans_delay, fl_download_trans_delay, sl_uplink_trans_delay, sl_download_trans_delay

    def cal_sl_activation_trans_delay(self,batch_lst = None):
        """
        :return: FL 上行时延、下行时延
                SL 上行时延，下行时延
        """
        if batch_lst is None:
            batch_lst = [len(i) for i in self.user_groups]
        flr, slr = [self.fl_band_ratio[i] * (1 - self.b0) if self.fl_lst[i] == 1 else 0 for i in
                    range(len(self.fl_lst))], \
                   [self.b0 if self.sl_lst[i] == 1 else 0 for i in range(len(self.sl_lst))]  # 在rate中将不在本簇的用户置0

        self.sl_uplink_cap =[self.cal_communication_capability(i, self.pku, slr[i] * self.Bandwidth) if self.sl_lst[i] == 1 else 0 for i
             in range(len(self.sl_lst))]
        self.sl_download_cap = \
            [self.cal_communication_capability(i, self.pkd, slr[i]* self.Bandwidth)
             if self.sl_lst[i] == 1 else 0 for i in range(self.args.num_users)]

        sl_activation_uplink_trans_delay, sl_activation_download_trans_delay,sl_other_download_trans_delay,sl_other_up_trans_delay = \
            [(self.activations[self.cutlayer_lst[i]][-1] * len(batch_lst)) / uc
             if self.sl_lst[i] != 0 else 0 for i, (uc, dc) in
             enumerate(zip(self.sl_uplink_cap, self.sl_download_cap))], \
            [(self.activations[self.cutlayer_lst[i]][-1] * len(batch_lst)) / dc
             if self.sl_lst[i] != 0 else 0 for i, (uc, dc) in
             enumerate(zip(self.sl_uplink_cap, self.sl_download_cap))], \
            [(self.model_param[self.cutlayer_lst[i]]) / uc
             if self.sl_lst[i] != 0 else 0 for i, (uc, dc) in
             enumerate(zip(self.sl_uplink_cap, self.sl_download_cap))], \
            [(self.model_param[self.cutlayer_lst[i]]) / dc
             if self.sl_lst[i] != 0 else 0 for i, (uc, dc) in
             enumerate(zip(self.sl_uplink_cap, self.sl_download_cap))]  # 每个客户端的通信时延,sl还有激活值的时延

        return sl_activation_uplink_trans_delay, sl_activation_download_trans_delay,sl_other_download_trans_delay,sl_other_up_trans_delay

    def cal_delay(self,batch_size_lst = None):
        fl_compute_delay, sl_compute_delay = self.cal_compute_delay(batch_size_lst)
        self.fl_compute_delay, self.sl_compute_delay = fl_compute_delay, sl_compute_delay
        fl_uplink_trans_delay, fl_download_trans_delay, sl_uplink_trans_delay, sl_download_trans_delay = self.cal_trans_delay(batch_size_lst)
        self.fl_uplink_trans_delay, self.fl_download_trans_delay, self.sl_uplink_trans_delay, self.sl_download_trans_delay = fl_uplink_trans_delay, fl_download_trans_delay, sl_uplink_trans_delay, sl_download_trans_delay

        fld, sld = [ut + dt + j for ut, dt, j in zip(fl_uplink_trans_delay, fl_download_trans_delay, fl_compute_delay)], \
                   [ut + dt + j for ut, dt, j in zip(sl_uplink_trans_delay, sl_download_trans_delay, sl_compute_delay)]
        self.fld,self.sld = fld,sld
        # 每个客户端的时延
        try:
            fld_mx, sld_s = max(fld), sum(sld)  # sl 是取总和，fl是取最大值
            return fld_mx, sld_s
        except Exception as e:
            print(e)
            input("错误")

    def cal_communication_capability(self, ind, p, bandwidth):
        """
        注意，广播信道不进行带宽的分配
        """
        # bandwidth = float(Bandwidth * band_ratio)
        return bandwidth * math.log2(1 + p * self.signal_cap[ind] / (bandwidth * self.noise))

    def batch_size_init(self):
        self.batch_size_lst = [1]*len(self.user_groups)
        self.lambda_lst = [0.1] *len(self.user_groups)
        self.mu = 0.1

    def batch_size_decision(self):
        # TODO rho2 暂定和rho1设定值相同
        # 如果这里的alpha_f很大，会导致第一次更新后lambda
        alpha_f = 0.00001
        alpha_s = 0.00001
        batch_min = 10


        fl_computation_delay, sl_computation_delay = self.cal_compute_delay(self.batch_size_lst)
        fl_up_trans_delay,fl_down_trans_delay,_,_ = self.cal_trans_delay(self.batch_size_lst)
        sl_activation_uplink_trans_delay, sl_activation_download_trans_delay,sl_other_download_trans_delay,sl_other_up_trans_delay = self.cal_sl_activation_trans_delay(self.batch_size_lst)
        # xi的系数
        # 对于FL computation_delay = xi*C(模型参数量) / f_{k,t} , Gamma = C(模型参数量) / f_{k,t}
        fl_Tau_lst = [d/b for d,b in zip(fl_computation_delay,self.batch_size_lst)]
        # 对于SL 同样,每个tau都需要除上对应的batch_size_lst
        sl_Tau_lst = [(a+b+c)/d for a,b,c,d in zip(sl_computation_delay,sl_activation_uplink_trans_delay,sl_activation_download_trans_delay,self.batch_size_lst)]

        # 初始化所有的用户的batch_size为1
        self.batch_size_init()
        # 计算时延的上界，就是所有设备都训练全部模型的情况下的时延
        tau_ub = [d*len(self.user_groups[i])/self.batch_size_lst[i] + fl_up_trans_delay[i]+fl_down_trans_delay[i] for i,d in enumerate(fl_computation_delay)]    # SL 的求和和 FL 单体的最大值
        tau_ub.append(sum([(a+b+c)*len(self.user_groups[i])/d+ sl_other_up_trans_delay[i]+sl_other_download_trans_delay[i]
                            for i,(a,b,c,d) in enumerate(zip(sl_computation_delay,sl_activation_uplink_trans_delay,sl_activation_download_trans_delay,self.batch_size_lst))]))
        tau_ub = max(tau_ub)
        tau_lb = [d / self.batch_size_lst[i] + fl_up_trans_delay[i] + fl_down_trans_delay[i]
                  for i, d in enumerate(fl_computation_delay)]  # SL 的求和和 FL 单体的最大值
        tau_lb.append(sum([(a + b + c)/ d + sl_other_up_trans_delay[i] +
                           sl_other_download_trans_delay[i]
                           for i, (a,b,c,d) in enumerate(
                zip(sl_computation_delay, sl_activation_uplink_trans_delay, sl_activation_download_trans_delay,
                    self.batch_size_lst))]))
        tau_lb = max(tau_lb)# SL 的求和和 FL 单体的最大值
        tau_star = 0

        tau_gap = 1 - sum(self.lambda_lst) - self.mu

        cnt = 0

        while abs(tau_gap) > 0.001:
            # 当前实际的tau值
            tau = max([d + fl_up_trans_delay[i] + fl_down_trans_delay[i] for i, d in enumerate(fl_computation_delay)]
                      + [sum([(a + b + c) + sl_other_up_trans_delay[i] +
                              sl_other_download_trans_delay[i]
                              for i, (a,b,c,d) in enumerate(
                    zip(sl_computation_delay, sl_activation_uplink_trans_delay, sl_activation_download_trans_delay,
                        self.batch_size_lst))])])  # SL 的求和和 FL 单体的最大值
            # 计算当前的 batchsize 大小
            for i in range(len(self.user_groups)):
                # 为了后续的正常训练，这里的batch_size最少是10
                    self.batch_size_lst[i] = max(batch_min,min(math.sqrt(self.rho2/self.lambda_lst[i]/fl_Tau_lst[i]),len(self.user_groups[i])*0.8)) \
                        if self.fl_lst[i]!=0 \
                        else max(batch_min,min(math.sqrt(self.rho2/self.mu/sl_Tau_lst[i]),len(self.user_groups[i])*0.8))

            tau_gap = 1 - sum(self.lambda_lst) - self.mu
            tau_star = tau if abs(tau_gap) < 0.00001 else (
                tau_ub if tau_gap>0 else tau_lb
            )

            self.lambda_lst = [max(0.00001,self.lambda_lst[i] + alpha_f*(fl_Tau_lst[i]*self.batch_size_lst[i]- tau_star)) if self.fl_lst[i]!=0 else 0 for i in range(self.user_num)]
            self.mu = max(0.00001,self.mu + alpha_s*(sum([fl_Tau_lst[i]*self.batch_size_lst[i] for i in range(self.user_num)]) - tau_star))

            alpha_s-=0.0000000000000000001
            alpha_f-=0.0000000000000000001

            # 防止陷入死循环，如果没有fl用户或者有fl用户但是最大的lambda就是0.00001（即所有lambda值都更新为最小值） 且 mu值都更新为最小值或者sl用户为空
            # 如果这里出现了一千次，可以认定为收敛了
            if ((max(self.lambda_lst)==0.00001 or sum(self.fl_lst)==0) and (self.mu == 0.00001 or sum(self.sl_lst)==0)) or  min(self.batch_size_lst)==batch_min or max(self.batch_size_lst) == max(self.user_groups):
                cnt+=1
                if cnt == 1000:
                    break

        # logger.debug("done")

