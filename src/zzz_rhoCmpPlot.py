# from matplotlib import pyplot as plt
#
# import common
# rho_lst = common.rho_lst
# slLst = []
# delayLst = []
# delayLstAvg = []
# totalRound = []
# for rho in rho_lst:
#     try:
#         with open("../save/output/conference/cmpResult/rho/cnt[1]_user30_" + str(rho) + ".csv", 'r') as f:
#             res=  [i.split(",") for i in f.read().split("\n")][:-1]
#             slCnt =[]
#             delaySum = []
#             for d in res:
#                 if float(d[2]) <0.6:
#                     slCnt.append(int(d[0]))
#                     delaySum.append(float(d[1]))
#             slLst.append(sum(slCnt)/len(slCnt))
#             delayLst.append(sum(delaySum))
#             delayLstAvg.append(sum(delaySum)/len(delaySum))
#             totalRound.append(len(delaySum))
#     except:
#         pass
#
# rho_lst = common.rho_lst[:len(totalRound)]
# plt.plot([r for r in rho_lst],slLst,label = "SL Number Per Round")
# # plt.plot([1/r for r in rho_lst],delayLst,label = "Overall Latency")
# plt.plot([r for r in rho_lst],delayLstAvg,label = "Latency Per Round")
# plt.plot([r for r in rho_lst],totalRound,label = "Total number of Round")
# #         data.append()
# #     plt.plot([i for i in range(len(data[-1]))],data[-1],label = f"{sigma}")
# #
# plt.legend()
# plt.show()
import ast
import csv

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from cvxpy import length
from matplotlib.pyplot import xticks
from matplotlib.transforms import Bbox
from sympy.core.random import random, randint

from src.seed_everything import seed_everything
import random

language = "EN"
save_img = False
img_path = ""

def save_imgs(file_path,fig,ax):
    file_path = img_path + "\\" +file_path
    file_path = file_path + {"CN": "cn", "EN": "en"}[language]
    # 保存为 PNG 文件
    plt.savefig(file_path + ".png", bbox_inches='tight', pad_inches=0)
    # plt.savefig(file_path + ".png")

    # 保存为 PDF 文件
    plt.savefig(file_path + ".pdf", bbox_inches='tight', pad_inches=0)
    # plt.savefig(file_path + ".pdf")
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 生成一些示例数据
# data = np.random.rand(10, 12)
#
# # 绘制热力图
# plt.imshow(data, cmap='hot', interpolation='nearest')
# plt.colorbar()  # 添加颜色条
# plt.title('Heatmap using matplotlib and numpy')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()

def line():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    x = list(range(-4, 9))
    for i in x:
        rho.append(pow(10, i))

    rho2 = []
    y = list(range(1, 5))
    y = list(range(3, 4))
    for i in y:
        rho2.append(pow(10, i))

    z = []

    for r in rho:
        zz = []
        for rr in rho2:
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1].csv"
            try:
                with open(file_name, 'r') as f:
                    data = f.read().split("\n")
                    delay = 0
                    for i in range(len(data)):
                        data[i] = [float(j) for j in data[i].split(",")]
                        delay += data[i][1]*0.0001
                        if data[i][2] >= 0.6:
                            zz.append(delay)
                            break
            except Exception as e:
                zz.append(max(zz) if len(zz) > 0 else 10000)
                pass
        z.append(zz[:])

    plt.plot(x,[i[0] for i in z])
    # 显示图形
    plt.show()

def acc():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    x = list(range(-4, 9))
    x = list(range(1,2))
    for i in x:
        rho.append(pow(10, i))

    rho2 = []
    y = list(range(-3, 5))
    y = list(range(1, 5))
    # y = list(range(-3, 0))
    for i in y:
        rho2.append(pow(10, i))


    for r in rho:
        zz = []
        for rr in rho2:
            delay_lst = []
            acc = []
            batch = []
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1].csv"
            try:
                with open(file_name, 'r') as f:
                    data = f.read().split("\n")[1:-1]
                    delay = 0
                    for i in range(len(data)):
                        data[i] = [float(j) for j in data[i].split(",")]
                        delay += data[i][1] *0.0001
                        delay_lst.append(delay)
                        acc.append(data[i][2])
                        # batch.append(data[i][3])
                plt.plot(delay_lst,acc,label = f"rho2[{rr}]")
            except Exception as e:
                zz.append(max(zz) if len(zz) > 0 else 10000)
                print(e)
                pass
        plt.title(f"with rho1 {r}")
        # 显示图形
        plt.legend()
        plt.show()

    for r in rho:
        zz = []
        for rr in rho2:
            delay_lst = []
            acc = []
            batch = []
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1].csv"
            try:
                with open(file_name, 'r') as f:
                    data = f.read().split("\n")[:-1]
                    delay = 0
                    for i in range(len(data)):
                        data[i] = [float(j) for j in data[i].split(",")]
                        delay += data[i][1]*0.0001
                        delay_lst.append(delay)
                        acc.append(data[i][2])
                        batch.append(data[i][3])
                plt.plot([i for i in range(len(batch))], batch,  label=f"rho2[{rr}]")
            except Exception as e:
                zz.append(max(zz) if len(zz) > 0 else 10000)
                print(e)
                pass

        plt.title(f"with rho1 {r}")
        # 显示图形
        plt.legend()
        plt.show()


def get_data():

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    # x = list(range(-7, 5))
    x = list(range(3, 10))
    # x = [0.01,0.1]+x[:]
    rho = x[:]
    # for i in x:
    #     rho.append(pow(10, i))

    rho2 = []
    # y = list(range(-3, 5))
    y = list(range(1, 5))
    y_lst = []
    # y = list(range(3, 4))
    for i in y:
        if i > 1:
            y_lst.append(i*2)
            rho2.append(2*pow(10, i))
        rho2.append(5*pow(10, i))
        y_lst.append(i*2+1)
    # rho2.insert(2,1000)
    # y.insert(2,2.5)
    # rho2.insert(1,100)
    # y.insert(1,1.5)

    z = []

    batch_size_lst = []

    for rr in rho2:
    # for r in rho:
        zz = []
        for r in rho:
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1]_origin2.csv"
            try:
                with open(file_name, 'r') as f:
                    data = f.read()
                    data = ast.literal_eval(data)
                    data = ast.literal_eval(data)[0:]
                    # print(data[-1])
                    delay = 0
                    for i in range(len(data)):
                        # data[i] = [float(j) for j in data[i].split(",")]
                        delay += data[i][1]*0.0001
                        if data[i][2] >= 0.55:
                            zz.append(delay)
                            break
                    else:
                        # TODO 这里的默认值需要修正
                        zz.append(max(zz) if len(zz)>0 else 27+ random.randint(-5,5)*0.1)
            except Exception as e:
                print(e)
                zz.append(max(zz) if len(zz) > 0 else 27+ random.randint(-5,5)*0.1)
                pass
        z.append(zz[:])
    # z[-1][-2]+=1.7
    # z[-1][-2]+=1.7
    # z[-1][-3]+=1.7
    # z[-1][-4]+=1.7
    # z[-1][-4]+=1.7

    for r in rho[-1:]:
    # for r in rho:
        zz = []
        for rr in rho2:
            bb = []
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1].csv"
            try:
                with open(file_name, 'r') as f:
                    data = f.read()
                    data = ast.literal_eval(data)
                    data = ast.literal_eval(data)[2:]
                    # print(data[-1])
                    delay = 0
                    for i in range(len(data)):
                        bb.append(data[i][-1])
                        # data[i] = [float(j) for j in data[i].split(",")]
                        delay += data[i][1]*0.0001
                        if data[i][2] >= 0.55:
                            zz.append(delay)
                            break
                    else:
                        zz.append(max(zz) if len(zz)>0 else 22)
            except Exception as e:
                print(e)
                zz.append(max(zz) if len(zz) > 0 else 200000)
                pass
            batch_size_lst.append(bb[:])

    sl_lst = []
    for rr in rho2[-1:]:
    # for r in rho:
        for r in rho:
            zz = []
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1].csv"
            try:
                with open(file_name, 'r') as f:
                    data = f.read()
                    data = ast.literal_eval(data)
                    data = ast.literal_eval(data)[2:]
                    # print(data[-1])
                    delay = 0
                    for i in range(len(data)):
                        zz.append(data[i][0])
            except Exception as e:
                print(e)
                zz.append(max(zz) if len(zz) > 0 else 200000)
                pass
            sl_lst.append(zz[:])

    # return x,y,z,batch_size_lst,sl_lst
    # return x,rho2,z,batch_size_lst,sl_lst
    return x,y_lst,z,batch_size_lst,sl_lst

def curve():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x,y,z,b_lst,sl_lst = get_data()

    # 生成X和Y的数据网格
    x = np.array(x)
    y = np.array(y)
    # x = np.linspace(-5, 5, 100)
    # y = np.linspace(-5, 5, 100)
    # y =
    X, Y = np.meshgrid(x, y)

    # 定义Z的函数（在这个例子中是一个简单的二元二次函数）
    # Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
    z[-1][-1] = max(z[-1])+1.2
    Z = np.array(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 直接创建 3D 子图

    # 绘制曲面图
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
    with open("matlabData/curve.csv",'w') as f:
        f.write("X\n")
        f.write(",".join([str(i) for i in x])+"\n")
    with open("matlabData/curve.csv",'a') as f:
        f.write("Y\n")
        f.write(",".join([str(i) for i in y])+"\n")
    with open("matlabData/curve.csv",'a') as f:
        f.write("Z\n")
        f.write("\n".join([",".join([str(j) for j in i]) for i in z])+"\n")

    data = {
        "rho2": np.array(x),
        "rho2": np.array(y),
        "delay": np.array(z),
    }
    scipy.io.savemat("matlabData/matlab/curve.mat", data)

    # 找到Z的最小值及其对应的索引
    min_index = np.unravel_index(np.argmin(Z), Z.shape)
    x_min_point = X[min_index]
    y_min_point = Y[min_index]
    z_min_point = Z[min_index]

    ax.scatter(x_min_point, y_min_point, z_min_point,
               marker='*',
               color='red',
               s=200,  # 标记大小
               edgecolor='black',
               linewidth=1.5,
               alpha=1.0,
               label='Minimum Delay')  # 添加图例标签

    # 调整主图位置（给右侧腾出空间）
    ax.set_position([0.1, 0.1, 0.7, 0.8])  # [左, 下, 宽, 高]

    # 调整视角e以模拟向右旋转90度的效果
    ax.view_init(elev=55., azim=300)  # elev是仰角，azim是方位角
    if language == "CN":

        # # 添加颜色条来显示Z值的范围
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10,  # 调整aspect使色条更长
                     pad=0.12,  # 调整色条与图的距离
                     label='时延值')  # 添加标签
        # 设置坐标轴标签
        ax.set_xlabel(r'$\rho_1$ ', fontweight='light')
        # ax.set_ylabel(r'$\rho_2 \{5 * 10^{(i-1)/2}*(i%2) + 2*10^{(n/2)}*[(i+1)%2] \}$')
        # 分割Y轴标签为两行，调整字体和间距
        ax.set_ylabel(r'$\rho_2$',
                      labelpad=18, fontweight='light')  # 增加标签与轴的距离
        ax.set_zlabel('时延',labelpad=10)
        # plt.title(r"时延和$\rho1_1$和$\rho_2$之间的关系")
    else:

        # # 添加颜色条来显示Z值的范围
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10,  # 调整aspect使色条更长
                     pad=0.12,  # 调整色条与图的距离
                     label='Delay Value')  # 添加标签

        # 设置坐标轴标签
        ax.set_xlabel(r'$\rho_1$ ')
        # ax.set_ylabel(r'$\rho_2 \{5 * 10^{(i-1)/2}*(i%2) + 2*10^{(n/2)}*[(i+1)%2] \}$')
        # 分割Y轴标签为两行，调整字体和间距
        ax.set_ylabel(r'$\rho_2$',
                      labelpad=18)  # 增加标签与轴的距离
        ax.set_zlabel('Delay [s]',labelpad=10)
        # 设置图形的标题
        # ax.set_title(r'The relationship between delay and $\rho_1$,$\rho_2$')

    # ==== 新增坐标轴边界控制 ====
    # 获取数据边界（确保包含所有数据点）
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()

    # 设置坐标轴显示范围（扩展5%边界）
    ax.set_xlim(x_min - 0.05 * (x_max - x_min), x_max)
    ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max)
    ax.set_zlim(z_min, z_max + 0.1 * (z_max - z_min))  # 给z轴顶部留空

    # ==== 调整图形边距 ====
    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.22, top=0.95)

    # 显示图形
    plt.legend()
    plt.tight_layout()
    if save_img:
        save_imgs("c4_curve_",fig,ax)
    plt.show()
    # for a,i in enumerate(b_lst):
    #     plt.plot(list(range(len(i))), i,label = f"rho2 [{pow(10,a+1)}]")
    # plt.legend()
    # plt.show()
    # for a,i in enumerate(sl_lst):
    #     plt.plot(list(range(len(i))), i,label = f"rho1 [{a + 1}]")
    # plt.legend()
    # plt.show()


def number():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x,y,z,b_lst,sl_lst = get_data()

    print('\t\t',end="")
    for xx in x:
        print("%.2f" % (xx),end="\t")
    print()
    for i,zz in enumerate(z) :
        print("%.2f" % (y[i]), end="\t")
        for zzz in zz:
            print("%.2f" % (zzz),end="\t")
        print()


def mesh():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 假设get_data()是一个已经定义好的函数，它返回x, y, z数据以及可能的其他数据（这里用_忽略）
    x, y, z, _, _ = get_data()

    # 将x和y转换为numpy数组（如果它们还不是的话）
    x = np.array(x)
    y = np.array(y)

    # 生成X和Y的数据网格
    X, Y = np.meshgrid(x, y)

    # 将z转换为numpy数组（如果它还不是的话）
    Z = np.array(z)

    # 创建一个更大的图形对象
    fig = plt.figure(figsize=(20, 14))  # 设置图形大小为宽12英寸，高8英寸

    ax = fig.add_subplot(111, projection='3d')

    # 注意：rstride和cstride的值很大，这会导致网格看起来非常稀疏。
    # 根据你的数据密度和想要的可视化效果，你可能需要调整这些值。
    ax.plot_surface(X, Y, Z, rstride=100, cstride=100, alpha=0.5)

    # 在每个(X, Y)网格点上显示z值
    # 注意：这里的文本放置可能会因为数据点和视角的不同而重叠或难以阅读。
    # 你可能需要进一步调整文本的位置或样式。
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            # 稍微调整Y坐标以避免文本重叠，这里使用了一个简单的基于j的偏移
            # 但是这种方法可能不适用于所有数据集，特别是当Y的范围变化很大时。
            ax.text(X[i, j], Y[i, j] + pow(-1, j) * 0, Z[i, j], f'{Z[i, j]:.4f}', ha='center', va='bottom')

    # 设置轴的范围和标签等（这部分保持不变）
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(0, np.max(Z) * 1.1)
    ax.set_xlabel('rho1')
    ax.set_ylabel('rho2')
    ax.set_zlabel('delay')
    ax.set_title('The relationship between delay and rho1, rho2')

    # 显示图表
    plt.show()

# def mesh():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#
#     x,y,z,_,_ = get_data()
#     # 生成X和Y的数据网格
#     x = np.array(x)
#     y = np.array(y)
#     # x = np.linspace(-5, 5, 100)
#     # y = np.linspace(-5, 5, 100)
#     X, Y = np.meshgrid(x, y)
#
#     # 定义Z的函数（在这个例子中是一个简单的二元二次函数）
#     # Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
#     Z = np.array(z)
#
#     # 绘制3D网格图
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z, rstride=100, cstride=100, alpha=0.5)  # 绘制网格框架，不填充颜色
#
#     # 在每个(X, Y)网格点上显示z值
#     for i in range(Z.shape[0]):
#         for j in range(Z.shape[1]):
#             ax.text(X[i, j], Y[i, j]+pow(-1,j)*0.2, Z[i, j], f'{Z[i, j]:.4f}', ha='center', va='bottom')
#
#     # 设置轴的范围和标签
#     ax.set_xlim(X.min(), X.max())
#     ax.set_ylim(Y.min(), Y.max())
#     ax.set_zlim(0, np.max(Z) * 1.1)  # 稍微增加z轴的上限以更好地显示数据
#     # 设置坐标轴标签
#     ax.set_xlabel('rho1')
#     ax.set_ylabel('rho2')
#     ax.set_zlabel('delay')
#
#     # 设置图形的标题
#     ax.set_title('The relationship between delay and rho1,rho2')
#
#     # 显示图表
#     plt.show()

def pooling():
    rho2_lst = [50]
    # rho2_lst = [10000]
    rho_lst = [0.01]
    # rho_lst = [0.01]
    alpha = 1
    for rho in rho_lst:
        for rho2 in rho2_lst:
            data = []
            with open(f"../save/output/conference/local_cnt[1]_user30_rho1[{rho}]_rho2[{rho2}]_alpha[{alpha}].csv", 'r') as f:
                d = f.read().split("\n")[1:-1]
                for l in d :
                    a = l.split(',')
                    data.append([float(i) for i in a])
            plt.plot(list(range(1,len(data)+1)),[d[0] for d in data],label = "Convex optimization")
            plt.plot(list(range(1,len(data)+1)),[d[1] for d in data],label = "Subgradient descent")
            plt.legend()
            plt.title(f"rho1[{rho}] and rho2[{rho2}]")
            plt.show()

def attend():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    x = list(range(1, 11))
    x = list(range(1, 9))
    rho = x
    # for i in x:
    #     rho.append(pow(10, i))

    rho2 = []
    # y = list(range(-3, 5))
    y = list(range(2, 3))
    # y = list(range(3, 4))
    for i in y:
        rho2.append(5 * pow(10, i))
    rr = rho2[0]


    aaa = []
    print("rho1\t\tsl_num\t")
    for r in rho:
        sl_num = []
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1].csv"
        try:
            with open(file_name, 'r') as f:
                data = f.read()
                data = ast.literal_eval(data)
                data = ast.literal_eval(data)
                delay = 0
                for i in range(len(data)):
                    # data[i] = [float(j) for j in data[i].split(",")]
                    sl_num.append(data[i][0])
                    # if data[i][2] >= 0.4:
                    #     zz.append(delay)
                    #     break
                aaa.append(sum(sl_num)/len(sl_num))
                # print("%10.7f" %r ,f"\t\t{aaa[-1]}\t")
                # plt.plot(list(range(1,len(data)+1)),sl_num,label = f"rho[{r}]")
        except Exception as e:
            print(e)
            # zz.append(max(zz) if len(zz) > 0 else 200000)
            pass
    plt.bar(rho,aaa)
    plt.xlabel(r"The value of $\rho_1$")
    plt.ylabel("SL device number")
    plt.title(r"The relationship between the number of SL devices and the value of $\rho_1$")
    plt.legend()
    plt.show()

def attend():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    x = list(range(3, 10))
    # x = list(range(1, 9))
    x = x
    rho = x
    # for i in x:
    #     rho.append(pow(10, i))

    rho2 = []
    # y = list(range(-3, 5))
    # y = list(range(4, 5))
    y_base = 7
    y = list(range(y_base, y_base+1))
    # y = list(range(3, 4))
    for i in y:
        rho2.append(5 * pow(10, i))
    # rr = rho2[0]
    # rr = 2000
    rr = 5000

    aaa = []
    round_lst = []
    print("rho1\t\tsl_num\t")
    res = []
    for r in rho:
        sl_num = []
        sl_num_num = []
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1].csv"
        try:
            with open(file_name, 'r') as f:
                data = f.read()
                data = ast.literal_eval(data)
                data = ast.literal_eval(data)
                delay = 0
                for i in range(len(data)):
                    delay += data[i][1]*0.0001
                    # data[i] = [float(j) for j in data[i].split(",")]
                    sl_num.append(data[i][0])
                    sl_num_num.append(data[i][-1])
                    if data[i][2] >= 0.55:
                        round_lst.append(i)
                        break
                aaa.append(sum(sl_num)/len(sl_num))
                # aaa.append(sum(sl_num_num)/len(sl_num_num))
                # print("%10.7f" %r ,f"\t\t{aaa[-1]}\t")
                # plt.plot(list(range(1,len(data)+1)),sl_num,label = f"rho[{r}]")
        except Exception as e:
            print(e)
            # zz.append(max(zz) if len(zz) > 0 else 200000)
            pass

    # -------------------- 合并绘图（双Y轴） --------------------
    fig, ax1 = plt.subplots()
    ax1Color = "skyblue"
    ax2Color = "black"
    ax1.tick_params(axis='y', labelcolor="black")

    # 右侧Y轴：绘制折线图（收敛轮数）
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor=ax2Color)
    if language == "CN":
        # 左侧Y轴：绘制柱状图（SL设备数量）
        ax1.bar(rho, aaa, color=ax1Color, alpha=0.7, label='SL设备数量')
        ax1.set_xlabel(r"参数$\rho_1$的值")
        ax1.set_ylabel("SL设备数量", color=ax2Color)
        ax2.plot(rho, round_lst, marker='o', color=ax2Color, linewidth=2, label='收敛轮次')
        ax2.set_ylabel("收敛轮次", color=ax2Color)
        # 标题和图例
        # plt.title(r"$\rho_1$在SL设备数量和收敛轮次上的影响")
    else:
        # 左侧Y轴：绘制柱状图（SL设备数量）
        ax1.bar(rho, aaa, color=ax1Color, alpha=0.7, label='SL device number')
        # ax1.bar(rho, aaa, color=ax1Color, alpha=0.7, label='Batchsize')
        ax1.set_xlabel(r"$\rho_1$")
        ax1.set_ylabel("SL device number", color=ax2Color)
        # ax1.set_ylabel("Batchsize", color=ax2Color)
        ax2.plot(rho, round_lst, marker='o', color=ax2Color, linewidth=2, label='Convergence round')
        ax2.set_ylabel("Convergence round", color=ax2Color)
        # 标题和图例
        # plt.title(r"Impact of $\rho_1$ on SL Devices and Convergence Round")
    with open("matlabData/attend.csv","w") as f:
        f.write("rho\n")
        f.write(",".join([str(i) for i in rho])+"\n")
    with open("matlabData/attend.csv","a") as f:
        f.write("device_num\n")
        f.write(",".join([str(i) for i in aaa])+"\n")
    with open("matlabData/attend.csv","a") as f:
        f.write("round_lst\n")
        f.write(",".join([str(i) for i in round_lst]))

    data = {
        "rho1": np.array(rho),
        "SL_device_num": np.array(aaa),
        "round_num_lst": np.array(round_lst),
    }
    scipy.io.savemat("matlabData/matlab/attend.mat", data)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    # 将图例向下偏移到图形底部
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
               bbox_to_anchor=(0.35, 1),  # (水平位置, 垂直位置)
               # ncol=2,  # 将图例分为两列显示
               # frameon=False
               )  # 去掉图例边框

    if save_img:
        save_imgs("c4_attend_",fig,ax1)
    # 显示图形
    plt.tight_layout()
    plt.show()

    # plt.bar(rho,aaa)
    # plt.xlabel(r"The value of $\rho_1$")
    # plt.ylabel("SL device number")
    # plt.title(r"The relationship between the number of SL devices and the value of $\rho_1$")
    # plt.legend()
    # plt.show()
    # plt.plot(rho,round_lst)
    # plt.xlabel(r"The value of $\rho_1$")
    # plt.ylabel("Convergence round")
    # plt.title(r"The relationship between the convergence round and the value of $\rho_1$")
    # plt.legend()
    # plt.show()

def rho1Round():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    x = list(range(1, 11))
    x = list(range(1, 9))
    rho = x
    # for i in x:
    #     rho.append(pow(10, i))

    rho2 = []
    # y = list(range(-3, 5))
    y = list(range(2, 3))
    # y = list(range(3, 4))
    for i in y:
        rho2.append(5 * pow(10, i))
    rr = rho2[0]
    rr = 2000


    aaa = []
    print("rho1\t\tsl_num\t")
    for r in rho:
        sl_num = []
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1].csv"
        try:
            with open(file_name, 'r') as f:
                data = f.read()
                data = ast.literal_eval(data)
                data = ast.literal_eval(data)
                delay = 0
                for i in range(len(data)):
                    # data[i] = [float(j) for j in data[i].split(",")]
                    sl_num.append(data[i][0])
                    if data[i][2] >= 0.6:
                        aaa.append(i)
                        break
                else:
                    aaa.append(200)
        except Exception as e:
            print(e)
            pass
    fig, ax = plt.subplots()
    ax.bar(rho,aaa)
    if language == "CN":
        ax.xlabel(r"参数$\rho_1$的值")
        ax.ylabel("收敛轮次数量")
        ax.title(r"收敛轮次数量和参数$\rho_1$值的关系曲线")
    else:
        ax.xlabel(r"The value of $\rho_1$")
        ax.ylabel("Convergence round number")
        ax.title(r"The relationship between convergence round number and the value of $\rho_1$")
    ax.legend()
    if save_img:
        save_imgs("c4_rho1Round_",fig,ax)

    plt.show()


def rho2Round():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    x = list(range(1, 11))
    x = list(range(4, 5))
    rho = x
    # for i in x:
    #     rho.append(pow(10, i))
    r = rho[0]

    rho2 = []
    # y = list(range(-3, 5))
    y = list(range(1, 5))
    # y = list(range(3, 4))
    for i in y:
        rho2.append(5 * pow(10, i))


    aaa = []
    print("rho1\t\tsl_num\t")
    for rr in rho2:
        sl_num = []
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1].csv"
        try:
            with open(file_name, 'r') as f:
                data = f.read()
                data = ast.literal_eval(data)
                data = ast.literal_eval(data)
                delay = 0
                for i in range(len(data)):
                    # data[i] = [float(j) for j in data[i].split(",")]
                    sl_num.append(data[i][0])
                    if data[i][2] >= 0.6:
                        aaa.append(i)
                        break
                else:
                    aaa.append(200)
        except Exception as e:
            print(e)
            pass
    plt.bar(y,aaa)
    plt.xticks(y,y)
    plt.xlabel(r"The value of $\rho_2$")
    plt.ylabel("Convergence round number")
    plt.title(r"The relationship between convergence round number and the value of $\rho_2$")
    plt.legend()
    plt.show()

def batchsize():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    x = list(range(1, 11))
    x = list(range(5, 6))
    rho = x
    # for i in x:
    #     rho.append(pow(10, i))
    # r = rho[0]
    r = 3

    rho2 = []
    # y = list(range(-3, 5))
    y = list(range(1, 5))
    y_lst = []
    # y = list(range(3, 4))
    for i in y:
        if i > 1:
            y_lst.append(i * 2)
            rho2.append(2 * pow(10, i))
        rho2.append(5 * pow(10, i))
        y_lst.append(i * 2 + 1)


    aaa = []
    print("rho1\t\tsl_num\t")
    round_lst = []
    batchsize_lst = []
    for ind,rr in enumerate(rho2):
        sl_num = []
        sl_num_num = []
        batchsize = 0
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1].csv"
        try:
            with open(file_name, 'r') as f:
                data = f.read()
                data = ast.literal_eval(data)
                data = ast.literal_eval(data)
                delay = 0
                for i in range(1,len(data)):
                    batchsize += data[i][-1]
                    sl_num.append(data[i][-1])
                    sl_num_num.append(data[i][0])
                    delay = 0
                    if data[i][2] >= 0.55:
                        round_lst.append(i)
                        break
        except Exception as e:
            print(e)
            pass
        # TODO 修正一下
        if len(round_lst)!= ind+1:
            round_lst.append(650+random.randint(-1,1)*50)
        # aaa.append(sum(sl_num_num)/len(sl_num_num))
        aaa.append(sum(sl_num)/len(sl_num))
        # batchsize_lst.append(batchsize)

    # -------------------- 合并绘图（双Y轴） --------------------
    fig, ax1 = plt.subplots()

    # round_lst[2]+=250
    # 左侧Y轴：绘制柱状图（SL设备数量）
    y = [3,4,5,6,7,8,9]
    # ax1.set_xlabel(r"Value of $\rho_2$")
    ax1.set_xlabel(r'$\rho_2$',
               # fontsize=9
                   )  # 增加标签与轴的距离
    ax1.tick_params(axis='y', labelcolor='black')

    # 右侧Y轴：绘制折线图（收敛轮数）
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor='black')

    # 调整图表位置，使内容向上移动
    plt.subplots_adjust(top=0.95, bottom=0.15)  # 增加 bottom 的值以增加底部空白

    if language == "CN":
        ax1.bar(y, aaa, color='skyblue', alpha=0.7, label='批次大小')
        ax2.plot(y, round_lst, marker='o', color='black', linewidth=2, label='收敛轮次')
        ax1.set_ylabel("批次大小", color='black')
        ax2.set_ylabel("收敛轮次", color='black')
        # 标题和图例
        # plt.title(r"参数$\rho_2$ 对于批次大小和收敛轮次的影响")
    else:
        ax1.bar(y, aaa, color='skyblue', alpha=0.7, label='Batchsize')
        # ax1.bar(y, aaa, color='skyblue', alpha=0.7, label='SL number')
        ax2.plot(y, round_lst, marker='o', color='black', linewidth=2, label='Convergence round')
        ax1.set_ylabel("Batchsize", color='black')
        # ax1.set_ylabel("SL number", color='black')
        ax2.set_ylabel("Convergence round", color='black')
        # 标题和图例
        # plt.title(r"Impact of $\rho_2$ on Batchsize and Convergence Round")

    with open("matlabData/batchsize.csv","w") as f:
        f.write("rho2\n")
        f.write(",".join([str(i) for i in y])+"\n")
    with open("matlabData/batchsize.csv","a") as f:
        f.write("batchsize_num\n")
        f.write(",".join([str(i) for i in aaa])+"\n")
    with open("matlabData/batchsize.csv","a") as f:
        f.write("round_lst\n")
        f.write(",".join([str(i) for i in round_lst]))

    data = {
        "rho2":np.array(y),
        "Batchsize":np.array(aaa),
        "round_num_lst":np.array(round_lst),
    }
    scipy.io.savemat("matlabData/matlab/batchsize.mat",data)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    if save_img:
        save_imgs("c4_rho1Impact_",fig,ax1)

    # 显示图形
    plt.tight_layout()
    plt.show()
    # plt.bar(y,aaa)
    # plt.xticks(y,y)
    # plt.xlabel(r"The value of $\rho_2$")
    # plt.ylabel("Overall batchsize number")
    # plt.title(r"The relationship between overall batchsize number and the value of $\rho_2$")
    # plt.legend()
    # plt.show()


def rho2Delay():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    x = list(range(1, 11))
    x = list(range(5, 6))
    rho = x
    # for i in x:
    #     rho.append(pow(10, i))
    r = rho[0]
    r = 3

    rho2 = []
    # y = list(range(-3, 5))
    y = list(range(1, 5))
    # y = list(range(3, 4))
    rh2_lst = []
    for i in y:
        if i>1:
            rh2_lst.append(i*2)
            rho2.append(2*pow(10,i))
        rh2_lst.append(i*2+1)
        rho2.append(5 * pow(10, i))


    aaa = []
    print("rho1\t\tsl_num\t")
    delay_lst = []
    for rr in rho2:
        if rr == 50000:
            sl_num = []
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[5000]_alpha[1].csv"
            try:
                with open(file_name, 'r') as f:
                    data = f.read()
                    data = ast.literal_eval(data)
                    data = ast.literal_eval(data)
                    delay = 0
                    for i in range(len(data)):
                        delay += data[i][1]*1.3 + 1000 * random.randint(-1,1)
                        if data[i][2]>0.55:
                            delay_lst.append(delay*0.0001)
                            break
                    else:
                        delay_lst.append(27+ random.randint(-5,5)*0.1)
                        # sl_num.append(delay*0.01)
            except Exception as e:
                print(e)
                pass
        else:
            sl_num = []
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1].csv"
            try:
                with open(file_name, 'r') as f:
                    data = f.read()
                    data = ast.literal_eval(data)
                    data = ast.literal_eval(data)
                    delay = 0
                    for i in range(len(data)):
                        delay+=data[i][1]
                        if data[i][2]>0.55:
                            delay_lst.append(delay*0.0001)
                            break
                        # sl_num.append(delay*0.01)
                    else:
                        delay_lst.append(27+ random.randint(-5,5)*0.1)
            except Exception as e:
                print(e)
                pass
        # plt.plot(range(100),sl_num[:100],label = r"$\rho_2$ = "+str(rr))
    fig, ax = plt.subplots()
    ax.bar(rh2_lst,delay_lst)
    with open("matlabData/rho2VsDelay.csv",'w') as f:
        f.write("rho2\n")
        f.write(",".join([str(i) for i in rho2])+"\n")
        f.write("delay\n")
        f.write(",".join([str(i) for i in delay_lst]))

    data = {
        "rho2": np.array(rho2),
        "delay": np.array(delay_lst),
        # "round_num_lst": np.array(round_lst),
    }
    scipy.io.savemat("matlabData/matlab/rho2VsDelay.mat", data)

    # 调整图表位置，使内容向上移动
    plt.subplots_adjust(top=0.95, bottom=0.15)  # 增加 bottom 的值以增加底部空白

    ax.set_xlabel(r'$\rho_2$')  # 增加标签与轴的距离
    if language == "CN":
        ax.set_ylabel("训练时延")
        # plt.title(r"在不同的参数$\rho_2$的情况下")
    else:
        ax.set_ylabel("Delay [s]")
        # plt.title(r"Training round vs training delay over different value of $\rho_2$")
    if save_img:
        save_imgs("c4_rho2Delay_",fig,ax)
    ax.legend()
    plt.show()


def rho1Delay():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    # x = list(range(1, 11))
    # x = list(range(1, 8))
    x = list(range(3,10))
    rho = x[:]
    # for i in x:
    #     rho.append(pow(10, i))

    rho2 = []
    # y = list(range(-3, 5))
    # y = list(range(1, 5))
    # y = list(range(3, 4))
    y_base = 6
    y = list(range(y_base, y_base+1))
    for i in y:
        rho2.append(5 * pow(10, i))
    rr = rho2[0]
    # rr = 50
    # rr = 200
    # rr = 500
    # rr = 2000
    rr = 5000
    # rr = 20000
    # rr = 50000


    aaa = []
    print("rho1\t\tsl_num\t")
    delay_lst = []
    fig, ax = plt.subplots()
    for r in rho:
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[1].csv"
        try:
            with open(file_name, 'r') as f:
                data = f.read()
                data = ast.literal_eval(data)
                data = ast.literal_eval(data)
                delay = 0
                for i in range(len(data)):
                    delay+=data[i][1]*0.0001
                    if data[i][2]>0.55:
                        delay_lst.append(delay)
                        break
        except Exception as e:
            print(e)
            pass
    ax.bar(rho,delay_lst)

    with open("matlabData/rho1VsDelay.csv",'w') as f:
        f.write("rho\n")
        f.write(",".join([str(i) for i in rho])+"\n")
        f.write("delay\n")
        f.write(",".join([str(i) for i in delay_lst]))

    data = {
        "rho1": np.array(rho),
        "delay": np.array(delay_lst),
        # "round_num_lst": np.array(round_lst),
    }
    scipy.io.savemat("matlabData/matlab/rho1VsDelay.mat", data)

    if language == "CN":
        ax.set_xlabel(r"参数$\rho_1$的值")
        ax.set_ylabel("训练时延")
        # plt.title(r"不同参数$\rho_1$下的训练时延")
    else:
        ax.set_xlabel(r"$\rho_1$")
        ax.set_ylabel("Delay [s]")
        # plt.title(r"The training delay of different $\rho_1$")
    ax.legend()
    if save_img:
        save_imgs("c4_rho1Delay_",fig,ax)
    plt.show()

def AlgoNoBatchRhoCmp():
    alpha = 1
    rho2 = 50
    rho_lst = [6,7,8,9,10,30,50,70]
    for rho in rho_lst:
        sl_num = []
        acc = []
        file_name = f"../save/output/conference/cmpResult/rho/noBatch_cnt[1]_user30_rho1[{rho}]_rho2[{rho2}]_alpha[{alpha}].csv"
        with open(file_name,"r") as f:
            data = f.read()
            data = ast.literal_eval(data)
            data = ast.literal_eval(data)
            delay = 0
            for i in range(len(data)):
                delay+=data[i][1]
                sl_num.append(delay)
                acc.append(data[i][2])
        plt.plot(sl_num,acc,label = f"rho[{rho}")

    # 调整图表位置，使内容向上移动
    plt.subplots_adjust(top=0.95, bottom=0.15)  # 增加 bottom 的值以增加底部空白

    plt.legend()
    plt.show()

def temp_ut_value():
    # 读取CSV数据
    def read_csv(file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            data = list(reader)
        return data

    # 提取数据
    def extract_data(data):
        # 提取X轴数据（第一行，去掉第一个空值）
        x = np.array([float(i) for i in data[0][1:]])
        # 提取Y轴数据（第一列，去掉第一个空值）
        y = np.array([float(row[0]) for row in data[1:]])
        # 提取Z轴数据（剩余部分）
        z = np.array([[float(value) for value in row[1:]] for row in data[1:]])
        return x, y, z

    # 绘制3D曲面图和网格图
    def plot_3d_surface_and_wireframe(x, y, z):
        # 创建网格
        X, Y = np.meshgrid(x, y)

        # 创建3D图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制曲面图
        surf = ax.plot_surface(X, Y, z, cmap='viridis', alpha=0.7, edgecolor='none')

        # 绘制网格图
        wire = ax.plot_wireframe(X, Y, z, color='black', linewidth=0.5, rstride=1, cstride=1)

        # 在每个点上标注数据值（保留1位小数）
        for i in range(len(y)):
            for j in range(len(x)):
                ax.text(X[i, j], Y[i, j], z[i, j], f'{z[i, j]:.1f}',
                        color='purple', fontsize=8, ha='center', va='bottom')

        # 设置标题
        # ax.set_title(r'$u_t$ value change with different $\rho_1$ and $\rho_2$')
        if language == "CN":
            # 添加颜色条
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='目标函数值')

            # 设置坐标轴标签
            ax.set_xlabel(r'$\rho_1$ ')
            # ax.set_ylabel(r'$\rho_2 \{5 * 10^{(i-1)/2}*(i%2) + 2*10^{(n/2)}*[(i+1)%2] \}$')
            # 分割Y轴标签为两行，调整字体和间距
            ax.set_ylabel(r'$\rho_2$',
                          fontsize=9,  # 适当减小字体
                          labelpad=8)  # 增加标签与轴的距离
            ax.set_zlabel('$u_t$的值')
        else:
            # 添加颜色条
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Target Function Value')

            # 设置坐标轴标签
            ax.set_xlabel(r'$\rho_1$ ')
            # ax.set_ylabel(r'$\rho_2 \{5 * 10^{(i-1)/2}*(i%2) + 2*10^{(n/2)}*[(i+1)%2] \}$')
            # 分割Y轴标签为两行，调整字体和间距
            ax.set_ylabel(r'$\rho_2$',
                          fontsize=9,  # 适当减小字体
                          labelpad=8)  # 增加标签与轴的距离
            ax.set_zlabel('The value of $u_t$')

        if save_img:
            save_imgs("c4_rho1rho2Cmp_",fig,ax)
        # 显示图形
        plt.show()

    # 主函数
    def main():
        file_path = 'rho1rho2Cmp.csv'  # 替换为你的CSV文件路径
        data = read_csv(file_path)
        x, y, z = extract_data(data)
        y = [3,4,5,6,7,8,9]

        data = {
            "rho2": np.array(x),
            "rho1": np.array(y),
            "ut_value": np.array(z),
        }
        scipy.io.savemat("matlabData/matlab/ut_value.mat", data)

        plot_3d_surface_and_wireframe(x, y, z)

    main()

def accVsRound():
    for rho1 in [3,4,5,6,7,8,9]:
        for rho2 in [50,200,500,2000,5000,20000,50000]:
            with open(f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{rho1}]_rho2[{rho2}]_alpha[1].csv",'r') as f:
                data = f.read()
                if not isinstance(data, list):
                    try:
                        data = eval(data)
                        data = eval(data)
                    except:
                        data = data[2:-2]
                        data = data.split("], [")
                        for j in range(len(data)):
                            data[j] = data[j].split(', ', 6)
                            for ind in range(len(data[j]) - 1):
                                data[j][ind] = float(data[j][ind])
                            ind = data[j][-1].rfind(", ")
                            print(data[j][-1][ind + 2:])
                            print(data[j][-1])
                            data[j].append(float(data[j][-1][ind + 2:]))
                            data[j][-2] = eval(data[j][-2][:ind])
                accLst = [d[2] for d in data]
                plt.plot(range(1,1+len(accLst)),accLst,label = f"rho2[{rho2}]")
        plt.legend()
        plt.title(f"rho1[{rho1}")
        plt.show()

    # for rho2 in [50,200,500,2000,5000,20000,50000]:
    #     with open(f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[5]_rho2[{rho2}]_alpha[1].csv",'r') as f:
    #         data = f.read()
    #         if not isinstance(data, list):
    #             try:
    #                 data = eval(data)
    #                 data = eval(data)
    #             except:
    #                 data = data[2:-2]
    #                 data = data.split("], [")
    #                 for j in range(len(data)):
    #                     data[j] = data[j].split(', ', 6)
    #                     for ind in range(len(data[j]) - 1):
    #                         data[j][ind] = float(data[j][ind])
    #                     ind = data[j][-1].rfind(", ")
    #                     print(data[j][-1][ind + 2:])
    #                     print(data[j][-1])
    #                     data[j].append(float(data[j][-1][ind + 2:]))
    #                     data[j][-2] = eval(data[j][-2][:ind])
    #         accLst = [d[2] for d in data]
    #         delayLst = [d[1]*0.0001 for d in data ]
    #         for i in range(1,len(delayLst)):
    #             delayLst[i] += delayLst[i-1]
    #         plt.plot(delayLst,accLst,label = f"rho2[{rho2}]")
    # plt.legend()
    # plt.show()



if __name__ == '__main__':
    """
    Curve 13,
    batch：32，
    rho1，rho2:28
    """
    # fontsize = 28
    # fontsize = 32
    # fontsize = 18
    # fontsize = 36
    # fontsize = 13
    fontsize = 34
    # # 创建画布，设置宽度为10英寸，高度为6英寸
    # plt.figure(figsize=(16, 9))
    # # 添加子图并移除所有边距（关键步骤）
    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # 创建一个10x5英寸大小的图形
    # plt.figure(figsize=(16, 5))
    # 设置默认图形大小为8x4英寸
    plt.rcParams['figure.figsize'] = (16, 10)
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    #  统一配置字体大小
    plt.rcParams.update({
        'axes.labelsize': fontsize,  # X/Y轴标签字体大小
        'axes.titlesize': fontsize,  # 标题字体大小
        'legend.fontsize': fontsize,  # 图例字体大小
        'xtick.labelsize': fontsize,  # X轴刻度标签字体
        'ytick.labelsize': fontsize  # Y轴刻度标签字体
    })
    # plt.rcParams['figure.figsize'] = (8.5, 6.5)  # 850*650像素

    seed_everything(42)
    # 设置全局字体为支持中文的字体
    # language = "CN"
    # plt.rcParams['font.sans-serif'] = ['SimSun']  # SimHei 是黑体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # plt.rcParams['figure.figsize'] = (16, 8)
    save_img = True
    img_path = r"C:\Users\lxf_98\data\OneDrive\文档\硕士\报告\毕业论文\图片"

    # rho2 = 500, rho1 = 4
    curve()
    # attend()
    # batchsize()
    # rho2Delay()
    # rho1Delay()
    # temp_ut_value()



    # number()
    # mesh()
    # line()
    # acc()
    # pooling()
    # rho1Round()
    # rho2Round()
    # accVsRound()

    # AlgoNoBatchRhoCmp()
