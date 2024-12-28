from cProfile import label

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import common

desc_2_ind = {
    "ut":2,
    "target function ut":2,
    "tau gap":1,
    "batch": 0,
    "device batch summation": 0,
}

rho2_lst = [0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000,500000000]
rho2_lst = [0.5,1,5,10,50,100,500,1000,5000,10000,50000]
# rho2_lst = [100000]

style = ["-", "--", ":", "-.", "-", "--", ":", "-."]
mark = ["o", "*", "^", ">", "s", "p", "D"]

def plot(desc="ut",end=100000,smooth=1):
    for i,rho2 in enumerate(rho2_lst):
        try:
            with open("../save/output/conference/cmpResult/sigma/batch/new_new_new_sigma_" + str(0.001) + f"rho2_{rho2}.csv", 'r') as f:
                data = f.read().split("---")

            data = [eval(i) for i in data]

            ut_lst = [d[desc_2_ind[desc]] for d in data[1:]]

            ut_lst = ut_lst[0:end]

            ut_lst = [sum(ut_lst[i:i + smooth]) / smooth for i in range(len(ut_lst) - smooth)]

            # fig, ax = plt.subplots()
            # plt.plot([i for i in range(len(ut_lst))], ut_lst,label=f"rho2={rho2}", linestyle = style[i%len(style)], marker = mark[i%len(mark)])
            plt.plot([i for i in range(len(ut_lst))], ut_lst,label=f"rho2={rho2}", linestyle = style[i%len(style)])
        except:
            pass
    # # 设置单个轴的浮点数精度
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.xlabel("iteration round")
    plt.ylabel(f"{desc}")
    plt.title(f"[Batch Decision Algorithm] {desc} VS iteration")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 设置浮点数精度为小数点后两位
    # plt.rcParams['axes.formatter.limits'] = (-4, 4)
    end = 100000
    plot("target function ut",end,1)
    plot("tau gap",end,1)
    plot("device batch summation",end,1)
