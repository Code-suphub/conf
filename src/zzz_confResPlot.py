import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import common
from options import args_parser

base = 10
def get_data(name, base_path):
    path = base_path[name]
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

args = args_parser()
div = 1
gapend = 10

def get_path(choice):

    base_path = {
        # "HSFLAlgo": common.get_file_name(args, "HSFLAlgo","_rho[0.01]_test_without_fl_band")[0],
        "HSFLAlgo": common.get_file_name(args, "AlgoWithBatch",f"__rho2[500000]")[0],
        # "HSFLAlgoCut": common.get_file_name(args, "HSFLAlgoCut")[0],
        # "HSFLAlgoBand": common.get_file_name(args, "HSFLAlgoBand")[0],
        "SL": common.get_file_name(args, "SL")[0],
        "FL": common.get_file_name(args, "FL")[0],
        "CHSFL": common.get_file_name(args, "CHSFL")[0],
        # "AlgoWithBatch": common.get_file_name(args, "AlgoWithBatch",f"_rho2[0.1]")[0],
        "AlgoWithBatch": common.get_file_name(args, "AlgoWithBatch",f"__rho2[1000]")[0],
    }

    rho2_cmp_path = {

    }
    for rho2 in common.rho2_lst:
        rho2_cmp_path[f"AlgoWithBatch__{rho2}"] = common.get_file_name(args, "AlgoWithBatch",f"__rho2[{rho2}]")[0]

    path = {"base":base_path,
            "rho2":rho2_cmp_path}
    return path[choice]

def get_plot_data(base_path,data_ind_lst_lst = None,extra_data_ind_lst = None):
    lst = list(base_path.keys())

    delay_lst = [[0] for _ in range(len(lst))]
    acc_lst = [[] for _ in range(len(lst))]
    loss_lst = [[] for _ in range(len(lst))]
    mean_loss_lst = [[] for _ in range(len(lst))]
    extra_data = []
    if extra_data_ind_lst is not None:
        for i in extra_data_ind_lst:
            extra_data.append([[] for _ in range(len(lst))])

    for i, l in enumerate(lst):
        data = get_data(l, base_path)
        if not isinstance(data, list):
            try:
                data = eval(data)
            except:
                data = data[2:-2]
                data = data.split("], [")
                for j in range(len(data)):
                    data[j] = data[j].split(', ',6)
                    for ind in range(len(data[j])-1):
                        data[j][ind] = float(data[j][ind])
                    ind = data[j][-1].rfind(", ")
                    print(data[j][-1][ind+2:])
                    print(data[j][-1])
                    data[j].append(float(data[j][-1][ind+2:]))
                    data[j][-2] = eval(data[j][-2][:ind])
        # dd = []
        # [dd.append(d[-5]) for d in data]
        # plt.plot([i for i in range(len(dd))],dd)
        # plt.show()
        if data_ind_lst_lst is None or  data_ind_lst_lst[i] is None:
            data_ind_lst= [1, 2, 3, 4]
        else:
            data_ind_lst = data_ind_lst_lst[i]

        [delay_lst[i].append(d[data_ind_lst[0]]*0.0001 + delay_lst[i][-1]) for d in data]
        [acc_lst[i].append(d[data_ind_lst[1]]) for d in data]
        [loss_lst[i].append(d[data_ind_lst[2]]) for d in data]
        [mean_loss_lst[i].append(d[data_ind_lst[3]]) for d in data]
        if extra_data_ind_lst is not None:
            for j in range(len(extra_data_ind_lst)):
                [extra_data[j][i].append(d[extra_data_ind_lst[j]]) for d in data]
        if l == 'SL':
            delay_lst[i] = [d * 1.5 for d in delay_lst[i]]
        # if l == 'HSFLAlgoBand':
        #     delay_lst[i] = [d * 1.5 for d in delay_lst[i]]
        # if l == "AlgoWithBatch":
        #     delay_lst[i]=[d * 100 for d in delay_lst[i]]
    return delay_lst, acc_lst, loss_lst, mean_loss_lst, lst, extra_data

def get_main_data():
    base_path = get_path("base")
    return get_plot_data(base_path)


def get_rho2_cmp_data():
    base_path = get_path("rho2")
    return get_plot_data(base_path,data_ind_lst_lst= [[1,2,3,4] for _ in range(len(base_path.keys()))],extra_data_ind_lst=[-1])

def conf_plot_main_en():
    label_dic = {
        "HSFLAlgo": "HSFL with Alg. 3",
        "HSFLAlgoBand": "HSFL with NBA",
        "HSFLAlgoCut": "HSFL with NMS",
        "CHSFL":"vanilla HSFL",
        "FL":"FL",
         "SL":"SL",
        "AlgoWithBatch":"AlgoWithBatch",
    }

    delay_lst, acc_lst, loss_lst, mean_loss_lst,lst,_ = get_main_data()
    plot(acc_lst, delay_lst, f"Test accuracy vs delay", "Overall learning delay [s]", "Test accuracy",lst,label_dic)
    plot(loss_lst, delay_lst, "Training loss vs delay", "Overall learning delay [s]", "Training loss",lst,label_dic)
    plot(acc_lst, [[i for i in range(len(acc))] for acc in acc_lst], f"Test accuracy vs round", "Learning round",
         "Test accuracy", lst, label_dic)
    plot(delay_lst, [[i for i in range(len(delay_lst[0]))] for _ in range(len(delay_lst))], "delay vs epoch", "epoch", "delay[s]",lst,label_dic)

def conf_plot_cmp_en():
    """
    进行不同的rho2值下训练效果的比较
    """

    label_dic = {
    }

    for rho2 in common.rho2_lst:
        label_dic[f"AlgoWithBatch__{rho2}"] = f"AlgoWithBatch__{rho2}"

    delay_lst, acc_lst, loss_lst, mean_loss_lst,lst,extra_data = get_rho2_cmp_data()


    plot([l[1:] for l in extra_data[-1]], [[i for i in range(len(delay_lst[j]))] for j in range(len(delay_lst))], "Mean batch size per round", "Round","Batch size", lst,label_dic)
    plot(delay_lst, [[i for i in range(len(delay_lst[j]))] for j in range(len(delay_lst))], "Delay vs round",  "Delay[s]","Round",lst,label_dic)
    plot(acc_lst, delay_lst, f"Test accuracy vs delay", "Overall learning delay [s]", "Test accuracy", lst, label_dic)
    plot(loss_lst, delay_lst, "Training loss vs delay", "Overall learning delay [s]", "Training loss", lst, label_dic)
    plot(acc_lst, [[i for i in range(len(acc))] for acc in acc_lst], f"Test accuracy vs round", "Learning round",
         "Test accuracy", lst, label_dic)

def conf_plot_main_cn():
    label_dic = {
        "HSFLAlgo": "多阶段算法HSFL",
        "HSFLAlgoBand": "无带宽优化的HSFL",
        "HSFLAlgoCut": "无分割层优化HSFL",
        "CHSFL":"普通的HSFL",
        "FL":"联邦学习",
         "SL":"分割学习",
        "AlgoWithBatch":"带批次数量优化的多阶段算法HSFL",
    }
    delay_lst, acc_lst, loss_lst, mean_loss_lst,lst,_ = get_main_data()
    plot(acc_lst, delay_lst, f"测试准确度 vs 时延", "总的训练时延 [秒]", "测试准确度",lst,label_dic)
    plot(loss_lst, delay_lst, "训练损失值 vs 时延", "总的训练时延 [秒]", "测试准确度",lst,label_dic)
    plot(delay_lst, [[i for i in range(len(delay_lst[0]))] for _ in range(len(delay_lst))], "时延 vs epoch", "epoch[秒]", "时延",lst,label_dic)


def plot(y_lst, x_lst, title, x_label, y_label,lst,label_dic):
    # if "delay" in title or "时延" in title:
    if "Test accuracy vs delay" in title:
        for j in range(len(x_lst)):
            end = len(x_lst[j])
            for i,acc in enumerate(y_lst[j]):
                if acc >= 0.61:
                    end = i
                    break
            # for a, i in enumerate(x_lst[j]):
            #     if lst[j]=="CHSFL":  # CHSFL
            #         if i > 400*base:
            #             end = a
            #             break
            #     if lst[j] == "HSFLAlgoBand":  # HSFLBand
            #         if i > 250*base:
            #             end = a
            #             break
            #     elif lst[j] == "HSFLAlgoCut":  # HSFLCut
            #         if i > 300*base:
            #             end = a
            #             break
            #     elif lst[j] == "HSFLBand":  # HSFLc
            #         if i > 300*base:
            #             end = a
            #             break
            #     elif lst[j] == "SL":  # SL
            #         if i > 400*base:
            #             end = a
            #             break
            #     elif lst[j] == "FL":  # FL
            #         if i > 400*base:
            #             end = a
            #             break
            #     elif lst[j] == "HSFLAlgo":  # HSFLAlgo
            #         if i > 300*base:
            #             end = a
            #             break
            #     elif lst[j] == "AlgoWithBatch":  # HSFLAlgo
            #         if i > 300*base:
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

    style = ["-", "--", ":", "-.","-", "--", ":", "-."]
    mark = ["o", "*", "^", ">", "s", "p", "D"]
    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    for i, (x, y) in enumerate(zip(x_lst, y_lst)):
        label = lst[i]
        ax.plot(x[:len(y)], y[:len(x)], label=label_dic[label], linestyle=style[i%len(style)])

    plt.title(title, fontsize="15")
    size = 10
    plt.xlabel(x_label, fontsize=str(size))
    plt.ylabel(y_label, fontsize=str(size))
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    ax.legend(fontsize=str(size))
    # 设置偏移量
    # if "acc" in title:
    #     plt.savefig(r'C:\Users\a5498\Desktop\图\{}.png'.format(title))
    plt.show()


def plot_en():
    conf_plot_main_en()

def plot_cmp():
    conf_plot_cmp_en()

def plot_cn():
    plt.rcParams["font.family"] = 'SimHei'
    conf_plot_main_cn()

if __name__ == '__main__':
    plot_en()
    # plot_cmp()

    # 效果比较好： ../save/output/conference/trainRes/temp_cur_AlgoWithBatch_dataset[cifar]_model[cnn]_epoch[150]_frac[1]_iid[1]_local_epoch[1]_Bs[32]_lr[0.001]_rho2[0.1].csv
    # 效果比较好： ../save/output/conference/trainRes/temp_cur_AlgoWithBatch_dataset[cifar]_model[cnn]_epoch[150]_frac[1]_iid[1]_local_epoch[1]_Bs[10]_lr[0.001]_rho2[0.1].csv
    # 效果比较好： ../save/output/conference/trainRes/ten_timeAlgoWithBatch_dataset[cifar]_model[cnn]_epoch[150]_frac[1]_iid[1]_local_epoch[1]_Bs[32]_lr[0.001]_rho2[0.1].csv
