import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from src import common
from src.options import args_parser


def get_data(name, base_path):
    path = "../"+base_path[name]
    with open(path, 'r') as f:
        data = json.load(f)
    return data

"""
CHSFL: 98
HSFLBand: 153
HSFLCut: 89
HSFL: 316
SL: 39
FL : 657
"""


def conf_plot_main():
    args = args_parser()
    div = 1
    gapend = 10
    base_path = {
        "HSFLAlgo": common.get_file_name(args, "HSFLAlgo","_rho[0.01]_test_")[0],
        "HSFLAlgoCut": common.get_file_name(args, "HSFLAlgoCut")[0],
        # "HSFLAlgoBand": common.get_file_name(args, "HSFLAlgoBand")[0],
        # "SL": common.get_file_name(args, "SL")[0],
        # "FL": common.get_file_name(args, "FL")[0],
        # "CHSFL": common.get_file_name(args, "CHSFL")[0],
        # "AlgoWithBatch": common.get_file_name(args, "AlgoWithBatch",f"_rho2[10]")[0],
    }
    lst = list(base_path.keys())

    delay_lst = [[0] for _ in range(len(lst))]
    acc_lst = [[] for _ in range(len(lst))]
    loss_lst = [[] for _ in range(len(lst))]
    mean_loss_lst = [[] for _ in range(len(lst))]
    for i, l in enumerate(lst):
        data = get_data(l, base_path)
        if not isinstance(data, list):
            data = eval(data)
        # dd = []
        # [dd.append(d[-5]) for d in data]
        # plt.plot([i for i in range(len(dd))],dd)
        # plt.show()

        [delay_lst[i].append(d[-5] + delay_lst[i][-1]) for d in data]
        [acc_lst[i].append(d[-4]) for d in data]
        [loss_lst[i].append(d[-3]) for d in data]
        [mean_loss_lst[i].append(d[-2]) for d in data]
        # if l == 'SL':
        #     delay_lst[i] = [d * 0.5 for d in delay_lst[i]]
        # if l == 'HSFLAlgoBand':
        #     delay_lst[i] = [d * 1.5 for d in delay_lst[i]]
        # if l == "AlgoWithBatch":
        #     delay_lst[i]=[d * 100 for d in delay_lst[i]]
    # plot(acc_lst, delay_lst, f"Test accuracy vs delay", "Overall learning delay [s]", "Test accuracy")
    # plot(loss_lst, delay_lst, "Training loss vs delay", "Overall learning delay [s]", "Training loss")
    plot(acc_lst, delay_lst, f"测试准确度 vs 时延", "总的训练时延 [秒]", "测试准确度",lst)
    plot(loss_lst, delay_lst, "训练损失值 vs 时延", "总的训练时延 [秒]", "测试准确度",lst)


def plot(y_lst, x_lst, title, x_label, y_label,lst):
    if "delay" in title or "时延" in title:
        # if "acc vs delay" in title:
        for j in range(len(x_lst)):
            end = len(x_lst[j])
            # for a, i in enumerate(x_lst[j]):
            #     if lst[j]=="CHSFL":  # CHSFL
            #         if i > 55:
            #             end = a
            #             break
            #     if lst[j] == "HSFLBand":  # HSFLBand
            #         if i > 60:
            #             end = a
            #             break
            #     elif lst[j] == "HSFLCut":  # HSFLCut
            #         if i > 37:
            #             end = a
            #             break
            #     elif lst[j] == "HSFLBand":  # HSFLc
            #         # if i > 100:
            #         #     end = a
            #         #     break
            #     elif lst[j] == "SL":  # SL
            #         if i > 100:
            #             end = a
            #             break
            #     elif lst[j] == "FL":  # FL
            #         if i > 125:
            #             end = a
            #             break

            x_lst[j] = x_lst[j][:end]
            y_lst[j] = y_lst[j][:end]
    # if "loss vs delay" in title:
    #     for j in range(len(x_lst)):
    #         end = len(x_lst[j])
    #         for a,i in enumerate(x_lst[j]):
    #             if j ==0: #CHSFL
    #                 if i>55:
    #                     end = a
    #                     break
    #             if j == 1 : # HSFLBand
    #                 if i>19:
    #                     end = a
    #                     break
    #             elif j== 2: # HSFLCut
    #                 if i>40:
    #                     end = a
    #                     break
    #             elif j== 3: # HSFL
    #                 if i>40:
    #                     end = a
    #                     break
    #             elif j== 4: # SL
    #                 if i>30:
    #                     end = a
    #                     break
    #             elif j== 5: # FL
    #                 if i>60:
    #                     end = a
    #                     break
    #
    #         x_lst[j]=x_lst[j][:end]
    #         y_lst[j] = y_lst[j][:end]
    # if x_lst is None:
    #     x_lst = [[i for i in range(len(y_lst[0]))] for _ in range(len(y_lst))]

    style = ["-", "--", ":", "-."]
    mark = ["o", "*", "^", ">", "s", "p", "D"]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    for i, (x, y) in enumerate(zip(x_lst, y_lst)):
        label = lst[i]
        # if i == 1:
        #     label = lst[3]
        # elif i == 3:
        #     label = lst[1]
        if label == 'HSFLAlgo':
            label = "HSFL with Alg. 3"
            # label = "多阶段算法HSFL"
            ax.plot(x[:len(y)], y, label=label, linestyle="--")
        elif label == "HSFLAlgoBand":
            label = "HSFL with NBA"
            # label = "无带宽优化的HSFL"
            ax.plot(x[:len(y)], y, label=label, linestyle="--")
        elif label == "HSFLAlgoCut":
            label = "HSFL with NMS"
            # label = "无分割层优化HSFL"
            ax.plot(x[:len(y)], y, label=label, linestyle="-.")
        elif label == "CHSFL":
            label = "vanilla HSFL"
            # label = "普通的HSFL"
            ax.plot(x[:len(y)], y, label=label, linestyle=":")
        elif label == "FL":
            # label = "联邦学习"
            ax.plot(x[:len(y)], y, label=label, linestyle="-")
        elif label == "SL":
            # label = "分割学习"
            # ax.plot(x[:len(y)], y, label=label, marker=mark[i])
            ax.plot(x[:len(y)], y, label=label)
        else:
            # x = x[1:]
            # y = y[1:]
            # x = [i*100 for i in x]
            ax.plot(x[:len(y)], y, label=label)

    # plt.title(title, fontsize="15")
    size = 20
    plt.xlabel(x_label, fontsize=str(size))
    plt.ylabel(y_label, fontsize=str(size))
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    ax.legend(fontsize=str(size))
    # 设置偏移量
    # if "acc" in title:
    #     plt.savefig(r'C:\Users\a5498\Desktop\图\{}.png'.format(title))
    plt.show()

if __name__ == '__main__':
    plt.rcParams["font.family"] = 'SimHei'
    conf_plot_main()