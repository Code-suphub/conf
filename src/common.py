import math
import random

import numpy as np
import torch

from models import *

file_base = "../save/output/conference/trainRes/"
file_mid = "_dataset[{}]_model[{}]_epoch[{}]_frac[{}]_iid[{}]_local_epoch[{}]_Bs[{}]_lr[{}]"
file_tail = ".csv"

epoch_map = {
    "SL" : 100,
    "FL":600,
    "CHSFL":200,
    "HSFLAlgo":150,
    "HSFLAlgoBand":150,
    "HSFLAlgoCut":150,
    "AlgoWithBatch":150
}

def get_file_name(args,file_type,extra=""):
    print("this is "+ file_type + " training ")
    args.epochs = epoch_map[file_type]
    file_type = "ten_time"+file_type
    file_name =  (file_base + file_type + file_mid + extra + file_tail). \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.lr)
    print("*" * 20 + file_name + "*" * 20)
    return file_name,args

def train_seed():
    seed = 124
    try:
        from pytorch_lightning import seed_everything
        seed_everything(seed)
    except Exception as e:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def model_get(args,train_dataset):
    tempModel,global_model = None,None
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

    return tempModel,global_model


def cal_epsilon(ut_dif):
    sigma = 0.00075
    try:
        epsilon = 1 / (1 + math.pow(math.e, ut_dif / sigma + 0.0000001))
    except:
        # print(ut_value_new, ut_value, ut_dif / sigma + 0.0000001)
        ut_dif = ut_dif / 10
        try:
            epsilon = 1 / (1 + math.pow(math.e, ut_dif / sigma + 0.0000001))
        except:
            # print(ut_value_new, ut_value, ut_dif / sigma + 0.0000001)
            ut_dif = ut_dif / 10
            try:
                epsilon = 1 / (1 + math.pow(math.e, ut_dif / sigma + 0.0000001))
            except:
                # print(ut_value_new, ut_value, ut_dif / sigma + 0.0000001)
                ut_dif = ut_dif / 10
                try:
                    epsilon = 1 / (1 + math.pow(math.e, ut_dif / sigma + 0.0000001))
                except:
                    # print(ut_value_new, ut_value, ut_dif / sigma + 0.0000001)
                    ut_dif = ut_dif / 10
                    try:
                        epsilon = 1 / (1 + math.pow(math.e, ut_dif / sigma + 0.0000001))
                    except:
                        # print(ut_value_new, ut_value, ut_dif / sigma + 0.0000001)
                        ut_dif = ut_dif / 10
                        try:
                            epsilon = 1 / (1 + math.pow(math.e, ut_dif / sigma + 0.0000001))
                        except:
                            # print(ut_value_new, ut_value, ut_dif / sigma + 0.0000001)
                            ut_dif = ut_dif / 10
                            try:
                                epsilon = 1 / (1 + math.pow(math.e, ut_dif / sigma + 0.0000001))
                            except:
                                # print(ut_value_new, ut_value, ut_dif / sigma + 0.0000001)
                                ut_dif = ut_dif / 10
                                epsilon = 1 / (1 + math.pow(math.e, ut_dif / sigma + 0.0000001))

    return epsilon

rho = 0.01
rho2 = 0.1
def get_rho():
    return rho,rho2

def generate_new_lst(fl_lst,sl_lst,args):
    # 进行新状态的生成，====没有问题====
    while True:
        randon_exchange_fl = random.randint(0, args.num_users - 1)
        randon_exchange_sl = random.randint(0, args.num_users - 1)
        if fl_lst[randon_exchange_fl] == 1 or sl_lst[randon_exchange_sl] == 1:  # 不允许两个都是0，否则是无效交换
            break
    if randon_exchange_sl == randon_exchange_fl:  # 同一个用户直接交换行为值
        fl_lst[randon_exchange_fl], sl_lst[randon_exchange_sl] = sl_lst[randon_exchange_sl], fl_lst[
            randon_exchange_fl]  # 簇间行为交换
    else:  # 不同用户需要判断，如果任何一个是1，那么就需要切换另一个用户的状态，并且这个切换行为不是互斥的
        if fl_lst[randon_exchange_fl] == 1:
            fl_lst[randon_exchange_fl], sl_lst[randon_exchange_fl] = 0, 1  # 这个用户到达另一个学习方式组
        if sl_lst[randon_exchange_sl] == 1:
            fl_lst[randon_exchange_fl], sl_lst[randon_exchange_fl] = 1, 0
    return fl_lst,sl_lst