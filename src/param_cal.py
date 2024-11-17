from typing import List

import numpy as np
import torch

# from models.load_model import load_model
from random_generate import *
from torchstat import stat

def Large_Scale_Generate(pos_rrh, pos_user, num_rrh):
    # Generate path loss
    num_user = 1  # In this simulation, only one typical user is considered
    dis_rrh_user = []
    for i in range(len(pos_rrh)):
        dis_rrh_user.append(
            [np.sqrt(np.square(np.real(pos_rrh[i] * np.ones((num_user, 1)) - pos_user)) +
                    np.square(np.imag(pos_rrh[i]  * np.ones((num_user, 1)) - pos_user))).item()]
        )
    dis_rrh_user = np.array(dis_rrh_user) # 注意，这里的dis_rrh_user中是一个列向量
    dis_rrh_user = 10 ** -3 * dis_rrh_user

    # Path loss is calculated according to Tony's paper
    path_loss_gain = 128.1 + 37.6 * np.log10(dis_rrh_user)

    # Antenna Gain 5 dBi
    antenna_gain = 5 * np.zeros((num_rrh, num_user))

    g_matrix_db = antenna_gain - path_loss_gain
    g_matrix = 10 ** (g_matrix_db / 10)

    return g_matrix


def Small_Scale_Generate(Num_RRH):
    # Only one user is considered in this simulation.
    H_Matrix = np.sqrt(1 / 2) * (np.random.randn(Num_RRH) + 1j * np.random.randn(Num_RRH))

    return H_Matrix


def cal_uplink_rate(n):
    """
    计算上行链路速率，单位是bit/s
    """
    Pos_RRH = pos_rand_generate(3*n)
    # Pos_RRH = [1000 + 1000j,100+100j,50+50j,2+4j,0.0001+0.0001j]
    Pos_User = [0 + 0j]
    Num_RRH = n
    p_k_u = 0.1  # 上传信道传输能量 0.1W
    bit_to_MB = 8e6

    Bandwidth = 1.4 * 10**6
    Noise = 10**(-134/10)

    Large_Scale_Matrix = Large_Scale_Generate(Pos_RRH, Pos_User, Num_RRH*3)
    Small_Scale_Matrix = Small_Scale_Generate(Num_RRH*3)
    # Small_Scale_Matrix = [-0.5996527 -1.14893061j, -0.58115789-0.03840965j -0.35488098+0.428154j  ]
    # print(Small_Scale_Matrix)
    capacity = [0]*(n)
    Noise_lst = []
    signal_cap = []

    for l in range(Num_RRH):
        Ch_Matrix = Small_Scale_Matrix[l] * np.sqrt(Large_Scale_Matrix[l])
        Noise_lst.append(Noise)
        signal_cap.append(np.abs(Ch_Matrix)**2*Bandwidth)
        SINR = Bandwidth * np.log2(1 + p_k_u*np.abs(Ch_Matrix)**2 / Noise)
        a = Bandwidth*math.log2(1+p_k_u*signal_cap[-1]/(Bandwidth*Noise))
        #
        capacity[l]=SINR.item()/32 # 因为上传模型参数或者是激活值都是float32类型的，所以这里直接进行转换
    for l in range(Num_RRH,Num_RRH*2):
        Ch_Matrix = Small_Scale_Matrix[l] * np.sqrt(Large_Scale_Matrix[l])
        signal_cap.append(np.abs(Ch_Matrix)**2*Bandwidth)

    return capacity,signal_cap,Bandwidth,Noise,p_k_u


# def cal_uplink_rate(n):
#     """
#     计算每个设备的上行链路传输速率，通过香农公式进行计算
#     """
#     pathloss = distance_rand_generate(n)
#     w = 1.4 * (10 ** 6)  # 频谱带宽，1.4MHz
#     p_k_u = 0.1  # 上传信道传输能量 0.1W
#     n0 = -174  # 噪声功率密度 -174dBm
#     h_k_u = 1
#
#     return [w * math.log(1 + ((p_k_u * h_k_u) / (w * n0)- p), 2) for p in pathloss]  # 上行链路传输速率


def cal_model_param(model) -> List:
    """
    计算模型每一层的参数量
    """
    total_params = 0
    dict1 = {}
    if False:
        for name, param in model.named_parameters():
            layer_params = param.numel()  # 当前层的参数量
            n = name.split('.')[0]
            dict1[n] = dict1.get(n, 0) + layer_params  # 将某一层的weight 和 bias 的数目进行相加
            total_params += layer_params

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.Conv2d) or isinstance(module, torch.nn.Linear):
            params = sum(param.numel() for param in module.parameters())
            total_params += params
            dict1[name] = params
        elif isinstance(module, torch.nn.modules.MaxPool2d):
            params = 0
            dict1[name] = params
    for d in dict1:
        print(f"{d} has {dict1[d]} params.")
    print(f"Total number of params: {total_params}")
    return list(dict1.values()) + [total_params]


def cal_model_activation(model,data) -> List:
    """
    计算模型每一层的激活值的数量，返回结果中包含每一层的in_channel、out_channel、w、h数，
        池化层in_channel、out_channel和上一层卷积层假定相同
        全连接层in_channel、out_channel设定为一
        其中in_channel对最后的计算没有作用
    """
    num_activations = []
    shape = list(data.shape) # 首先获取输入图片的维度.倒数第二个是宽，倒数第一个是高
    w = shape[-2]
    h = shape[-1]
    in_c= 1
    out_c = 1
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            # 如果该层是卷积层，则计算激活元数量
            in_c = layer.in_channels
            out_c = layer.out_channels
            kernel_size = layer.kernel_size[0]
            pad = layer.padding[0]
            stride = layer.stride[0]
            w_out = int((w-kernel_size+2*pad)/stride)+1  # 输出维度计算 (w_in -kernel_size+ 2*padding)/stride +1
            h_out = int((h-kernel_size+2*pad)/stride)+1
            w,h = w_out,h_out
            num_activations.append((in_c,out_c,w,h,out_c*w*h))
        elif isinstance(layer, torch.nn.modules.Linear):
            in_c= 1
            out_c = 1
            h=1
            w = layer.out_features
            num_activations.append((in_c,out_c,w,h,out_c*w*h))
        elif isinstance(layer,torch.nn.modules.MaxPool2d):
            w,h=w/2,h/2
            num_activations.pop()
            num_activations.append((in_c,out_c,w,h,out_c*w*h))

    # 输出每一层激活元数量
    # for name, (w,h,in_c,out_c,total) in num_activations.items():
    #     print("{}层  w:{} h:{} in_c:{} out_c:{} total_activation:{}".format(name, w,h,in_c,out_c,total))

    return num_activations


def cal_model_flops(model,data):
    """
    通过stat库计算模型参数和flops
    """
    shape = tuple(data.shape) # 这里需要传入训练数据除了batch维度之外其他的三个维度
    if len(shape)>3:
        shape = shape[-3:]
    print(shape)
    res = stat(model,shape)
    res = res.split('\n')[1:-8] # 最后几行是多余数据
    print(res)
    layers = []
    for r in res:
        if 'pool' in r:
            continue
        layers.append(r.split(' '))
    for l in layers:
        for i in range(len(l)-1,-1,-1):
            if len(l[i]) == 0:
                del l[i]

    params = [layer[-8] for layer in layers]
    params = [float(p.replace(',', '')) for p  in params]
    flops = [layer[-5] for layer in layers]
    flops = [float(p.replace(',', '')) for p  in flops]
    for i in range(1,len(params)-1):
        params[i]+=params[i-1]
        flops[i]+=flops[i-1]
    print(params)
    return params,flops


if __name__ == '__main__':
    # cal_uplink_rate(20)
    # a = [1,2,3]
    # # b = np.array(a)
    # # print(10/b)
    # # print(a[-0])
    # d = torch.randn(1,1,28,28)
    # model,_ = load_model(model_name="MNIST")
    # print(model)
    # # print(model)
    # # # cal_model_param(model)
    # print(cal_model_activation(model, d))
    # # # model(d)
    # # # print(d.item())
    # # date = d.reshape(-1,d.shape[2],d.shape[3]).tolist() # stat需要 三位列表，将原数据进行维度变换后并转化为列表
    # # shape = tuple(d.shape[1:])
    # # print(shape)
    # cal_model_flops(model,d)
    pass
