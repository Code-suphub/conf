from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import common

desc_2_ind = {
    "ut":2,
    "target function ut":2,
    "tau gap":1,
    "batch": 0,
}

def plot(desc="ut",end=100000,smooth=1):
    with open("../save/output/conference/cmpResult/sigma/batch/new_sigma_" + str(0.001) + ".csv", 'r') as f:
        data = f.read().split("---")

    data = [eval(i) for i in data]

    ut_lst = [d[desc_2_ind[desc]] for d in data[1:]]

    ut_lst = ut_lst[0:end]

    ut_lst = [sum(ut_lst[i:i + smooth]) / smooth for i in range(len(ut_lst) - smooth)]

    fig, ax = plt.subplots()
    ax.plot([i for i in range(len(ut_lst))], ut_lst)

    # 设置单个轴的浮点数精度
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.xlabel("iteration round")
    plt.ylabel(f"{desc}")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 设置浮点数精度为小数点后两位
    # plt.rcParams['axes.formatter.limits'] = (-4, 4)
    end = 100000
    plot("target function ut",end,1)
    plot("tau gap",end,1)
    plot("batch",end,1)
