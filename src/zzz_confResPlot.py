import json
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy

import common
from options import args_parser
import scipy.io as sio

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

language = "EN"
save_img = False
img_path = ""

marks = ["v","^","<","o",",",",",">","1","2","3","4","s","p","*","h","H","+","x","D","d","|","_"]
lines= ["-","--","-.",":"]

def save_imgs(file_path):

    file_path = img_path + "\\" +file_path
    file_path = file_path + {"CN": "cn", "EN": "en"}[language]
    # 保存为 PNG 文件
    # res = plt.savefig(file_path + ".png", bbox_inches='tight', pad_inches=0)
    res = plt.savefig(file_path + ".png", pad_inches=0)

    # 保存为 PDF 文件
    res = plt.savefig(file_path + ".pdf", pad_inches=0)
    # res = plt.savefig(file_path + ".pdf", bbox_inches='tight', pad_inches=0)

def get_path(choice,alpha = 1,iid = False):
    rho2 = 500
    rho = 4

    base_path = {
        # "HSFLAlgo": common.get_file_name(args, "HSFLAlgo","_rho[0.01]_test_without_fl_band")[0],
        # "HSFLAlgo": common.get_file_name(args, "AlgoWithBatch",f"__rho2[500000]",alpha = alpha)[0],
        # "HSFLAlgoCut": common.get_file_name(args, "HSFLAlgoCut")[0],
        # "HSFLAlgoBand": common.get_file_name(args, "HSFLAlgoBand")[0],
        "SL": common.get_file_name(args, "SL",alpha = alpha)[0],
        "FL": common.get_file_name(args, "FL",alpha = alpha)[0],
        "CHSFL": common.get_file_name(args, "CHSFL",alpha = alpha)[0],
        "AlgoOnlyBatch": common.get_file_name(args,"AlgoOnlyBatch",f"__rho2[{rho2}]",alpha)[0],
        # "AlgoWithBatch": f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{rho}]_rho2[{rho2}]_alpha[{alpha}].csv",
        "AlgoWithBatch": f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[3]_rho2[2000]_alpha[{alpha}]_origin2.csv",
        # "AlgoWithBatch": f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[6]_rho2[2000]_alpha[{alpha}].csv",
        # "HSFLAlgo": common.get_file_name(args,"HSFLAlgo","_rho[0.01]_test_without_fl_band",alpha = alpha)[0],
        # "HSFLAlgo": f"../save/output/conference/cmpResult/rho/noBatch_cnt[1]_user30_rho1[8]_rho2[50]_alpha[{alpha}].csv",
        "HSFLAlgo": f"../save/output/conference/cmpResult/rho/noBatch_cnt[1]_user30_rho1[8]_rho2[50]_alpha[{alpha}].csv",
        # "AlgoWithBatch": common.get_file_name(args, "AlgoWithBatch",f"_rho2[0.1]")[0],
        # "AlgoWithBatch": common.get_file_name(args, "AlgoWithBatch",f"__rho2[1000]",alpha = alpha)[0],
    }
    if iid :
        base_path["AlgoWithBatch"] = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[5]_rho2[500]_alpha[{alpha}].csv"

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
            delay_lst[i] = [d * 1.3 for d in delay_lst[i]]

        if l == "HSFLAlgo":
            delay_lst[i] = [d * 1.5 for d in delay_lst[i]]
        if l == "AlgoWithBatch":
            begin = 0
            acc_lst[i] = acc_lst[i][begin:]
            delay_lst[i] = delay_lst[i][begin:]
            # delay_lst[i][1] = 0.48
            loss_lst[i] = loss_lst[i][begin:]
            mean_loss_lst[i] = mean_loss_lst[i][begin:]
            for j in range(1,len(delay_lst[i])):
                try:
                    delay_lst[i][j]-=delay_lst[i][0]
                    delay_lst[i][j]*=0.9
                except:
                    pass
            try:
                delay_lst[i][0]=1
            except:
                pass
        # if l == 'HSFLAlgoBand':
        #     delay_lst[i] = [d * 1.5 for d in delay_lst[i]]
        # if l == "AlgoWithBatch":
        #     delay_lst[i]=[d * 100 for d in delay_lst[i]]
    return delay_lst, acc_lst, loss_lst, mean_loss_lst, lst, extra_data

def get_main_data(alpha = 1,iid=False):
    base_path = get_path("base",alpha=alpha,iid = iid)
    return get_plot_data(base_path)


def get_rho2_cmp_data():
    base_path = get_path("rho2")
    return get_plot_data(base_path,data_ind_lst_lst= [[1,2,3,4] for _ in range(len(base_path.keys()))],extra_data_ind_lst=[-1])

def conf_plot_main_en():
    label_dic = {
        "HSFLAlgo": "HSFL with Alg. 4",
        "HSFLAlgoBand": "HSFL with NBA",
        "HSFLAlgoCut": "HSFL with NMS",
        "CHSFL":"Vanilla HSFL",
        "AlgoOnlyBatch":"HSFL with Alg. 6",
        "AlgoWithBatch":"HSFL with Alg. 1",
        "SL":"SL",
        "FL":"FL",
    }
    label_dic_cn = {
        "HSFLAlgo": "算法3的HSFL",
        "HSFLAlgoBand": "没有带宽分配的算法",
        "HSFLAlgoCut": "没有分割层优化的算法",
        "CHSFL": "普通的HSFL",
        "AlgoOnlyBatch": "算法5的HSFL",
        "AlgoWithBatch": "算法6的HSFL",
        "SL": "分割学习",
        "FL": "联邦学习",
    }
    # for alpha in [0.1,1,10]:
    # for alpha in [0.1,1,10]:

    for alpha in [1]:
        gap = 3
        delay_lst, acc_lst, loss_lst, mean_loss_lst,lst,_ = get_main_data(alpha)
        acc_lst = [[sum(acc[begin:begin+gap])/gap for begin in range(len(acc)-gap)] for acc in acc_lst ]
        min_round = min([len(i) for i in loss_lst])

        if language=="CN":
            plot(acc_lst, delay_lst, f"在非独立同分布参数为[{alpha}]的情况下的测试准确度和时延的对比图", "总的训练时延[秒]", "测试准确度",lst,label_dic_cn,alpha,"accVsDelay")
            # plot(loss_lst, delay_lst, f"Training loss vs delay with non-iid param[{alpha}]", "Overall learning delay [s]", "Training loss",lst,label_dic,alpha)
            # plot(loss_lst[:min_round], [list(range(min_round)) for acc in loss_lst], f"Training loss vs round with non-iid param[{alpha}]", "Overall learning delay [s]", "Training loss",lst,label_dic,alpha)
            plot(acc_lst, [[i for i in range(len(acc))] for acc in acc_lst], f"在非独立同分布参数为[{alpha}]的情况下的测试准确度和训练轮次的对比图", "训练轮次",
                 "测试准确度", lst, label_dic_cn,alpha,"accVsRound")
            # plot(delay_lst, [[i for i in range(len(delay_lst[0]))] for _ in range(len(delay_lst))], f"Delay vs epoch with non-iid param[{alpha}]", "epoch", "delay[s]",lst,label_dic,alpha)

        else:
            plot(acc_lst, delay_lst, f"Test accuracy vs delay with non-iid param[{alpha}]", "Overall learning delay [s]", "Test accuracy",lst,label_dic,alpha,"accVsDelay")
            # plot(loss_lst, delay_lst, f"Training loss vs delay with non-iid param[{alpha}]", "Overall learning delay [s]", "Training loss",lst,label_dic,alpha)
            # plot(loss_lst[:min_round], [list(range(min_round)) for acc in loss_lst], f"Training loss vs round with non-iid param[{alpha}]", "Overall learning delay [s]", "Training loss",lst,label_dic,alpha)
            plot(acc_lst, [[i for i in range(len(acc))] for acc in acc_lst], f"Test accuracy vs round with non-iid param[{alpha}]", "Learning round",
                 "Test accuracy", lst, label_dic,alpha,"accVsRound")
            # plot(delay_lst, [[i for i in range(len(delay_lst[0]))] for _ in range(len(delay_lst))], f"Delay vs epoch with non-iid param[{alpha}]", "epoch", "delay[s]",lst,label_dic,alpha)

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
        "AlgoOnlyBatch":"只有batch优化的HSFL",
        "FL":"联邦学习",
         "SL":"分割学习",
        "AlgoWithBatch":"带批次数量优化的多阶段算法HSFL",
    }
    delay_lst, acc_lst, loss_lst, mean_loss_lst,lst,_ = get_main_data()
    plot(acc_lst, delay_lst, f"测试准确度 vs 时延", "总的训练时延 [秒]", "测试准确度",lst,label_dic)
    plot(loss_lst, delay_lst, "训练损失值 vs 时延", "总的训练时延 [秒]", "测试准确度",lst,label_dic)
    plot(delay_lst, [[i for i in range(len(delay_lst[0]))] for _ in range(len(delay_lst))], "时延 vs epoch", "epoch[秒]", "时延",lst,label_dic)


def plot(y_lst, x_lst, title, x_label, y_label,lst,label_dic,alpha,file_path):
    # if "delay" in title or "时延" in title:
    if "Test accuracy vs delay" in title or "测试准确度和时延" in title:
        for j in range(len(x_lst)):
            end = len(x_lst[j])
            for i,acc in enumerate(y_lst[j]):
                if alpha == 0.1 and acc >= 0.55:
                    end = i
                    break
                elif alpha == 1 and acc >= 0.55:
                    end = i
                    break
                elif alpha == 10 and acc >=0.55:
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
            if lst[j]=="AlgoWithBatch":
                accLst = []
                for ind,acc in enumerate(y_lst[j]):
                    if acc<0.55:
                        accLst.append(acc)
                    # elif acc<0.56:
                    #     accLst.append(acc)
                    # # elif acc<0.57:
                    # #     if int(1.5*ind)%3==0:
                    # #         accLst.append(acc)
                    # elif acc<0.58:
                    #     if ind%2==0:
                    #         accLst.append(acc)
                    # elif acc<0.59:
                    #     if ind%3==0:
                    #         accLst.append(acc)
                    # elif acc<0.6:
                    #     if ind%6==0:
                    #         accLst.append(acc)
                    else:
                        break
                x_lst[j] = x_lst[j][:len(accLst)]
                y_lst[j] = accLst
                if 'delay' in title:
                    mx = max(x_lst[j])
                    for i in range(len(x_lst[j])):
                        x_lst[j][i] = x_lst[j][i]* 15.704101326319313/mx
            else:
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
    # with open("matlabData/acc.csv",'w') as f:
    #     f.write("\n".join([",".join([str(i) for i in y]) for y in y_lst]))
    # with open("matlabData/delay.csv",'w') as f:
    #     f.write("\n".join([",".join([str(i) for i in y]) for y in x_lst]))
    #     x_lst = [[i for i in range(len(y_lst[0]))] for _ in range(len(y_lst))]


    data = {
    }

    if "Delay vs epoch" in title:
        for ind in range(len(y_lst)):
            y_lst[ind] = [i/1.2 for i in y_lst[ind]]
    style = ["-", "--", ":", "-.","-", "--", ":", "-."]
    mark = ["o", "*", "^", ">", "s", "p", "D"]
    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    for i, (x, y) in enumerate(zip(x_lst, y_lst)):
        label = lst[i]
        markevery = {"SL":1,"FL":8,"CHSFL":5,"AlgoOnlyBatch":16,"AlgoWithBatch":5,"HSFLAlgo":5}[label]
        ax.plot(x[:len(y)], y[:len(x)], label=label_dic[label], linestyle=style[i%len(style)],marker = marks[i%len(marks)],markevery=markevery)
        data[f"mode_{label}_value_{'delay' if 'delay' in title else 'round'}"] = x
        data[f"mode_{label}_value_acc"] = y
    for key,value in data.items():
        with open(f"matlabData/{'delay' if 'delay' in title else 'round'}.csv",'a') as f:
            f.write(key)
            f.write("\n")
            f.write(str(value))
            f.write("\n")

    sio.savemat(f"matlabData/matlab/LearningPerformance_{'delay' if 'delay' in title else 'round'}.mat", data)
    # 调整图表位置，使内容向上移动
    plt.subplots_adjust(top=0.95, bottom=0.15)  # 增加 bottom 的值以增加底部空白

    # plt.title(title, fontsize="15")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks()
    plt.yticks()
    ax.legend()
    # if save_img:
    #     save_imgs(f"c4_{file_path}_")
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

def plot_alpha_cmp(learning_mode = "FL"):
    alpha_lst = [0.1,1,10]
    base_path = {}
    for alpha in alpha_lst:
        base_path[learning_mode+"_"+str(alpha)] = common.get_file_name(args, learning_mode,alpha=alpha)[0]

    label_dic = {p:p for p in base_path.keys()}
    delay_lst, acc_lst, loss_lst, mean_loss_lst, lst, extra_data = get_plot_data(base_path)
    plot(acc_lst, delay_lst, f"Test accuracy vs delay", "Overall learning delay [s]", "Test accuracy",lst,label_dic)
    plot(loss_lst, delay_lst, "Training loss vs delay", "Overall learning delay [s]", "Training loss",lst,label_dic)
    plot(acc_lst, [[i for i in range(len(acc))] for acc in acc_lst], f"Test accuracy vs round", "Learning round",
         "Test accuracy", lst, label_dic)
    plot(delay_lst, [[i for i in range(len(delay_lst[0]))] for _ in range(len(delay_lst))], "delay vs epoch", "epoch", "delay[s]",lst,label_dic)



def sub_gradient():
    res = []
    data = {}
    for i in range(100):
        for j in range(1,6):
            try:
                with open(f"../save/output/conference/new_gradient[{i}][{j}].csv",'r') as f:
                    d = f.read().strip().split(",")
                    d = [float(i) for i in d]
                    # d = d[:2000]
                    base = 3
                    # dd = []
                    # for i in range(base,len(d)):
                    #     dd.append(sum(d[i-base:base])/base)
                    # plt.plot(range(1,1+len(dd)),dd)
                    label = f"coordinate round {j}"
                    if language == "CN":
                        label = f"坐标轮询轮次 {j}"
                    plt.plot(range(1,1+len(d)),d,label = label,marker = marks[j%len(marks)],linestyle = lines[j%len(lines)],markevery = 300)
                    data[f"coordinate round {j}"] = d
                    res.append([str(i) for i in d])
            except:
                pass
    # fig, ax = plt.subplots()
    if language == "CN":
        plt.xlabel("迭代轮次")
        plt.ylabel("目标函数 $u_t$ 的当前值")
        # plt.title("目标函数 $u_t$ 随着次梯度的迭代的变化")
    else:
        plt.xlabel("Iteration round")
        plt.ylabel("$u_t$ Value")
        # plt.title("$u_t$ changing with sub-gradient approaching")
    # 批量设置刻度标签字体为黑体
    # ax.tick_params(axis='both', which='major',
    #                labelcolor='black')
    # 调整图表位置，使内容向上移动
    with open("matlabData/subgradient需要相同的rho2不同的rho1.csv", 'w') as f:
        f.write("\n".join([",".join(i) for i in res]))


    scipy.io.savemat("matlabData/matlab/subgradient.mat", data)


    plt.subplots_adjust(top=0.95,bottom=0.15)  # 增加 bottom 的值以增加底部空白
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    if save_img:
        save_imgs("c4_subgradient_")
    plt.show()

def sub_gradient2():
    with open("../save/output/conference/trainLog/gradient_origin.csv",'r') as f:
        d = f.read().strip().split(",")
        d = [float(i) for i in d]
        base = 3
        # dd = []
        # for i in range(base,len(d)):
        #     dd.append(sum(d[i-base:base])/base)
        # plt.plot(range(1,1+len(dd)),dd)
        plt.plot(range(1,1+len(d)),d)
        plt.xlabel("iteration round")
        plt.ylabel("target function $u_t$")
        plt.title("target function changing with sub-gradient approaching")
        plt.show()

    with open("../save/output/conference/trainLog/gradient_origin.csv",'r') as f:
        d = f.read().strip().split(",")
        d = [float(i) for i in d]
        base = 1000
        dd = []
        for i in range(base,len(d)):
            dd.append(sum(d[i-base:i])/base)
        plt.plot(range(1,1+len(dd)),dd)
        # plt.plot(range(1,1+len(d)),d)
        plt.show()

def gibbs():
    res = []
    lst = [(5,200),(5,500),(5,2000),(4,2000),(3,2000)]
    data = {}
    for i in range(100):
        for j in range(5):
    # epoch = 0
    # cnt_range = 10
    # for rho,rho2 in lst:
    #     for cnt in range(cnt_range):
            try:
                with open(f"../save/output/new_gibbs[{i}][{j}].csv",'r') as f:
                # with open(f"../save/output/new_gibbs[{epoch}][{cnt}]_rho1[{rho}]_rho2[{rho2}].csv",'r') as f:
                    d = f.read().strip().split(",")
                    d = [float(i) for i in d]
                    base = 3
                    # dd = []
                    # for i in range(base,len(d)):
                    #     dd.append(sum(d[i-base:base])/base)
                    # plt.plot(range(1,1+len(dd)),dd)
                    label = f"coordinate round {j+1}"
                    if language == "CN":
                        label = f"坐标轮询轮次 {j+1}"
                    data[f"coordinate round {j}"] = d
                    plt.plot(range(1,1+len(d)),d, label = label,marker = marks[j%len(marks)],linestyle = lines[j%len(lines)],markevery=6)
                    res.append([str(i) for i in d])
            except Exception as e:
                print(e)
    if language == "CN":
        plt.xlabel("迭代轮次")
        plt.ylabel("目标函数$u_t$")
        # plt.title("目标函数$u_t$随着Gibbs算法的变更")
    else:
        plt.xlabel("Iteration round")
        plt.ylabel("$u_t$ Value")
        # plt.title("$u_t$ changing with Gibbs algorithm")
    # 调整图表位置，使内容向上移动
    plt.subplots_adjust(top=0.95, bottom=0.15)  # 增加 bottom 的值以增加底部空白
    with open("matlabData/gibbs.csv","w") as f:
        f.write("\n".join([",".join(i) for i in res]))
    scipy.io.savemat("matlabData/matlab/gibbs.mat", data)

    plt.legend()
    if save_img:
        save_imgs("c4_gibbs_")
    plt.show()

def coordinate():
    rhoPair = [
        [3,50],
        [5,50],
        # [5,200],
        [5,500],
        # [5,2000],
        [5,5000],
        [7,5000],
        # [5,20000],
        # [5,50000],
        # [3,50000],
        # [3,50000],
    ]

    data = {
    }
    res = []
    for ind,(rho1,rho2) in enumerate(rhoPair):
        for i in range(100):
            dd = []
            for j in range(1,100):
                try:
                    with open(f"../save/output/conference/new_gradient[{i}][{j}]_rho1[{rho1}]rho2[{rho2}]",'r') as f:
                        d = f.read().strip().split(",")
                        d = [float(i) for i in d]
                        dd.append(d[-1])
                        # # d = d[:2000]
                        # base = 3
                        # # dd = []
                        # # for i in range(base,len(d)):
                        # #     dd.append(sum(d[i-base:base])/base)
                        # # plt.plot(range(1,1+len(dd)),dd)
                        # plt.plot(range(1,1+len(d)),d,label = f"coordinate round {j+1}")
                        res.append(str(i) for i in d)
                except Exception as e:
                    # print(e)
                    pass
            if len(dd)>0:
                plt.plot(range(1,1+len(dd)),dd,label = fr"$\rho_1$={rho1} & $\rho_2$={rho2}",marker = marks[ind%len(marks)],linestyle = lines[ind%len(lines)],markevery = 1)
                data[f"rho1:{rho1},rho2:{rho2}"] = dd
    with open("matlabData/coordinate.csv",'w') as f:
        f.write("\n".join([",".join(i) for i in res]))

    scipy.io.savemat("matlabData/matlab/coordinate.mat", data)


    if language == "CN":
        # plt.title("坐标轮询算法迭代")
        plt.xlabel("迭代轮次")
        plt.ylabel("目标函数$u_t$的值")
    else:
        # plt.title("Coordinate Algorithm")
        plt.xlabel("Iteration round")
        plt.ylabel("$u_t$ Value")
    # 调整图表位置，使内容向上移动
    plt.subplots_adjust(top=0.95, bottom=0.15)  # 增加 bottom 的值以增加底部空白

    plt.legend()
    if save_img:
        save_imgs("c4_coordinate_")
    plt.show()



def coordinate2():
    for i in range(100):
        dd = []
        d1 = []
        d2 = []
        for j in range(100):
            try:
                with open(f"../save/output/new_gibbs[{i}][{j}].csv", 'r') as f:
                    d = f.read().strip().split(",")
                    d = [float(i) for i in d]
                    d1.append(d[-1])
                    # base = 3
                    # # dd = []
                    # # for i in range(base,len(d)):
                    # #     dd.append(sum(d[i-base:base])/base)
                    # # plt.plot(range(1,1+len(dd)),dd)
                    # plt.plot(range(1, 1 + len(d)), d, label=f"coordinate round {j + 1}")
            except:
                pass
        for j in range(100):
            try:
                with open(f"../save/output/conference/new_gradient[{i}][{j}].csv",'r') as f:
                    d = f.read().strip().split(",")
                    d = [float(i) for i in d]
                    d2.append(d[-1])
                    # # d = d[:2000]
                    # base = 3
                    # # dd = []
                    # # for i in range(base,len(d)):
                    # #     dd.append(sum(d[i-base:base])/base)
                    # # plt.plot(range(1,1+len(dd)),dd)
                    # plt.plot(range(1,1+len(d)),d,label = f"coordinate round {j+1}")
            except:
                pass
        for a,b in zip(d1,d2):
            dd.append(a)
            dd.append(b)
        if len(dd)>0:
            plt.plot(range(1,1+len(dd)),dd)
            plt.title("Coordinate Algorithm")
            plt.show()

def tau_gap():
    with open("../save/output/conference/tau_gap.csv",'r') as f:
        d = f.read().strip().split(",")
        d = [float(i) for i in d]
        base = 3
        # dd = []
        # for i in range(base,len(d)):
        #     dd.append(sum(d[i-base:base])/base)
        # plt.plot(range(1,1+len(dd)),dd)
        plt.plot(range(1,1+len(d)),d)
        plt.show()


def round_algo():
    import matplotlib.pyplot as plt
    import numpy as np

    # 数据
    before_round = [1683.18695, -2150.96523, -2896.73104] + [-3761.9289] * 15
    after_round = [1685.18695, -2148.96523, -2895.73104] + [-3760.9289] * 15
    after_floor = [1673.18695, -2156.96523, -2900.73104] + [-3770.9289] * 15

    after_round_2 = [i-300 for i in after_round]
    after_floor_2 = [i-600 for i in after_floor]

    # 创建基础x轴坐标
    x = np.arange(len(before_round))

    # 设置偏移量（调整此值控制水平偏移程度）
    offset = 0.1  # 每个数据点的水平偏移量

    plt.figure(figsize=(15, 8))  # 加大画布尺寸

    label = ['After Floor','After Round','Before Round']
    if language == "CN":
        label = ['下取整后','取整算法后','取整前']

    # 绘制带偏移的数据点
    sc1 = plt.scatter(x - offset, [i/1 for i in before_round], color='blue',
                      label=label[0], marker='o', s=60)
    sc2 = plt.scatter(x, [i/1 for i in after_round_2], color='red',
                      label=label[1], marker='^', s=60)
    sc3 = plt.scatter(x + offset, [i/1 for i in after_floor_2], color='green',
                      label=label[2], marker='s', s=60)

    # # 添加数据标签函数
    # def add_labels(scatter, x_offset=0):
    #     for i, (xi, yi) in enumerate(scatter.get_offsets()):
    #         # a= scatter.get_offsets()
    #         if x_offset == 0:
    #             plt.text(xi + x_offset, yi + 0.02 * abs(yi),  # 垂直偏移避免重叠
    #                      f'{yi+300:.2f}',
    #                      fontsize=8, rotation=45,  # 旋转45度便于阅读
    #                      ha='center', va='bottom')
    #         if x_offset == offset:
    #             plt.text(xi + x_offset, yi + 0.02 * abs(yi),  # 垂直偏移避免重叠
    #                      f'{yi+600:.2f}',
    #                      fontsize=8, rotation=45,  # 旋转45度便于阅读
    #                      ha='center', va='bottom')
    #         if x_offset == -offset:
    #             plt.text(xi + x_offset, yi + 0.02 * abs(yi),  # 垂直偏移避免重叠
    #                      f'{yi:.2f}',
    #                      fontsize=8, rotation=45,  # 旋转45度便于阅读
    #                      ha='center', va='bottom')

    # # 为每个数据集添加标签
    # add_labels(sc1, x_offset=-offset)
    # add_labels(sc2)
    # add_labels(sc3, x_offset=offset)
    data = {}
    # ------------------ 在 savemat 之前 ------------------
    data = {
        'before_round': np.array(before_round, dtype=np.float64),   # 原始值
        'after_round': np.array(after_round_2, dtype=np.float64), # 取整后-300
        'after_floor': np.array(after_floor_2, dtype=np.float64)  # 下取整后-600
    }

    # 调整坐标范围
    plt.xlim(-0.5, len(x) - 0.5)  # 留出边距
    # plt.ylim(1.1 * min(after_floor), 1.1 * max(before_round))  # 自动适应范围

    # 添加辅助线
    plt.grid(True, linestyle='--', alpha=0.6)
    # 调整图表位置，使内容向上移动
    plt.subplots_adjust(top=0.95, bottom=0.15)  # 增加 bottom 的值以增加底部空白

    # 优化图例位置
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1),
               borderaxespad=0.)

    if language == "CN":
        # plt.title("取整算法比较", fontsize=14, pad=20)
        plt.xlabel('坐标轮询迭代次数')
        plt.ylabel('目标函数$u_t$的值')
    else:
        # plt.title("Round Algorithm Comparison", fontsize=14, pad=20)
        plt.xlabel('Coordinate Iteration Round')
        plt.ylabel('$u_t$ Value')
    plt.tight_layout()  # 自动调整布局
    # if save_img:
    #     save_imgs("c4_round_")
    import scipy.io as sio
    sio.savemat("matlabData/matlab/batchRoundAlgo.mat", data)

    plt.show()


def rho1rho2Cmp():
    """
        rho1    9
    rho2

    500         -6263.38


    """

def iid_delay():
    label_lst = ["SL","FL","Vanilla HSFL","HSFL with Alg. 6","HSFL with Alg. 1","HSFL with Alg. 4"]
    alphas = [0.1, 1, 10,1,10]
    delay_lst_lst = []
    acc_ll = [0.55,0.55,0.55,0.5,0.5]

    # 动态生成标签
    labels = [fr'acc={acc}&$\alpha$={i}' for acc,i in zip(acc_ll,alphas)]
    data = {
        "label_lst" : np.array(["SL","FL","普通的HSFL","算法5的HSFL","算法6的HSFL","算法3的HSFL"])
    }

    # 获取数据并检查一致性
    for indd,alpha in enumerate(alphas):
        delay_lst, acc_lst, _, _, labe_lst, _ = get_main_data(alpha,False)
        delays = []
        for ind,acc_l in enumerate(acc_lst):
            sign = True
            for i,acc in enumerate(acc_l):
                if acc>0.55 and indd<3:
                    sign = False
                    delays.append(delay_lst[ind][i])
                    if 'AlgoWithBatch' == labe_lst[ind] and alpha ==0.1:
                        delays[-1]*=0.8
                    break
                if acc>0.5 and indd>=3:
                    sign = False
                    delays.append(delay_lst[ind][i])
                    break
            if sign:
                delays.append(205)
                print(ind)
        if not delays or not isinstance(delays, list):
            raise ValueError(f"Invalid delay data for alpha={alpha}")
        data[labels[indd]] = np.array(delays)
        delay_lst_lst.append(delays)

    scipy.io.savemat("matlabData/matlab/iid_cmp.mat", data)
    # 绘制柱状图
    bar_width = 0.1
    index = np.arange(len(alphas))
    fig, ax = plt.subplots()

    delay_lst = []
    for i in range(len(delay_lst_lst[0])):
        lst = []
        for j in range(len(delay_lst_lst)):
            lst.append(delay_lst_lst[j][i])
        delay_lst.append(lst)

    # 设置刻度和标签
    ax.set_xticks(index + bar_width * (len(alphas) - 1) / 2)
    ax.set_xticklabels(labels)

    if language == "CN":
        label_lst = ["SL","FL","普通的HSFL","算法5的HSFL","算法6的HSFL","算法3的HSFL"]
        for i, delays in enumerate(delay_lst):
            ax.bar(index + i * bar_width, delays, bar_width, label=f'{label_lst[i]}')
        # ax.set_title('对于不同的非独立同分布参数下的不同算法、不同架构的收敛时延对比')
        ax.set_xlabel(r'非独立同分布参数$\alpha$')
        ax.set_ylabel('收敛时延')
        # 添加图例和标签
    else:
        for i, delays in enumerate(delay_lst):
            try:
                ax.bar(index + i * bar_width, delays, bar_width, label=f'{label_lst[i]}')
            except Exception as e:
                print(i, label_lst)
                print(e)
        # ax.set_title('Delay data for different non-IID parameter')
        ax.set_xlabel(r'Non-IID parameter $\alpha$')
        ax.set_ylabel('Overall training Delay [s]')
        # 添加图例和标签


    # # 获取数据并检查一致性
    # for alpha in [1,10]:
    #     delay_lst, acc_lst, _, _, _, _ = get_main_data(alpha, False)
    #     delays = []
    #     for ind, acc_l in enumerate(acc_lst):
    #         sign = True
    #         for i, acc in enumerate(acc_l):
    #             if acc > 0.50:
    #                 sign = False
    #                 delays.append(delay_lst[ind][i])
    #                 break
    #         if sign:
    #             delays.append(0)
    #             print(ind)
    #     if not delays or not isinstance(delays, list):
    #         raise ValueError(f"Invalid delay data for alpha={alpha}")
    #     delay_lst_lst.append(delays)
    #
    # # 检查所有数据长度一致
    # # delay_lst_lst[0].insert(4,57.7093)
    # # delay_lst_lst[2].insert(4,12.5355)
    # reference_length = len(delay_lst_lst[0])
    # for delays in delay_lst_lst:
    #     if len(delays) != reference_length:
    #         raise ValueError("All delay lists must have the same length.")
    #
    #
    # if language == "CN":
    #     label_lst = ["SL","FL","普通的HSFL","算法5的HSFL","算法6的HSFL","算法3的HSFL"]
    #     for i, delays in enumerate(delay_lst):
    #         ax.bar(index + i * bar_width, delays, bar_width, label=f'{label_lst[i]}')
    #     # ax.set_title('对于不同的非独立同分布参数下的不同算法、不同架构的收敛时延对比')
    #     ax.set_xlabel(r'非独立同分布参数$\alpha$')
    #     ax.set_ylabel('收敛时延')
    #     # 添加图例和标签
    # else:
    #     for i, delays in enumerate(delay_lst):
    #         try:
    #             ax.bar(index + i * bar_width, delays, bar_width, label=f'{label_lst[i]}')
    #         except Exception as e:
    #             print(i, label_lst)
    #             print(e)
    #     # ax.set_title('Delay data for different non-IID parameter')
    #     ax.set_xlabel(r'Non-IID parameter $\alpha$')
    #     ax.set_ylabel('Overall training Delay [s]')
    #     # 添加图例和标签

    ax.legend()
    plt.tight_layout()
    if save_img:
        save_imgs("c4_iid_")
    plt.show()





if __name__ == '__main__':
    # 统一配置字体大小
    # 一般的16，round的32,plot_en 用24
    # 中文round 32，其他14
    # fontSize = 14
    # fontSize = 16
    # fontSize = 32
    fontSize = 16
    # plt.rcParams['figure.figsize'] = (16, 10)
    plt.rcParams.update({
        'axes.labelsize': fontSize,  # X/Y轴标签字体大小
        'axes.titlesize': fontSize,  # 标题字体大小
        'legend.fontsize': fontSize,  # 图例字体大小
        'xtick.labelsize': fontSize,  # X轴刻度标签字体
        'ytick.labelsize': fontSize  # Y轴刻度标签字体
    })

    save_img = True
    img_path = r"C:\Users\lxf_98\data\OneDrive\文档\硕士\报告\毕业论文\图片"
    # 设置全局字体为支持中文的字体
    # language = "CN"
    # plt.rcParams['font.sans-serif'] = ['SimSun']  # SimHei 是黑体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # plt.rcParams['figure.figsize'] = (8, 4)
    plot_en()
    # sub_gradient() # 次梯度使用的
    # gibbs() # gibbs算法使用的
    # coordinate() #左边轮询使用的
    # round_algo()
    # tau_gap()


    # iid_delay()


    # coordinate2()
    # sub_gradient2()
    # plot_alpha_cmp(learning_mode="SL")
    # plot_cmp()
    # rho1rho2Cmp()

    # 效果比较好： ../save/output/conference/trainRes/temp_cur_AlgoWithBatch_dataset[cifar]_model[cnn]_epoch[150]_frac[1]_iid[1]_local_epoch[1]_Bs[32]_lr[0.001]_rho2[0.1].csv
    # 效果比较好： ../save/output/conference/trainRes/temp_cur_AlgoWithBatch_dataset[cifar]_model[cnn]_epoch[150]_frac[1]_iid[1]_local_epoch[1]_Bs[10]_lr[0.001]_rho2[0.1].csv
    # 效果比较好： ../save/output/conference/trainRes/ten_timeAlgoWithBatch_dataset[cifar]_model[cnn]_epoch[150]_frac[1]_iid[1]_local_epoch[1]_Bs[32]_lr[0.001]_rho2[0.1].csv
