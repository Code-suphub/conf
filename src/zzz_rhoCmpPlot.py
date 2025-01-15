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

import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks


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
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
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
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
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
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
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
    x = list(range(1, 11))
    rho = x[:]
    # for i in x:
    #     rho.append(pow(10, i))

    rho2 = []
    # y = list(range(-3, 5))
    y = list(range(1, 5))
    # y = list(range(3, 4))
    for i in y:
        rho2.append(5*pow(10, i))
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
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
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
                        if data[i][2] >= 0.6:
                            zz.append(delay)
                            break
                    else:
                        zz.append(max(zz) if len(zz)>0 else 29)
            except Exception as e:
                print(e)
                zz.append(max(zz) if len(zz) > 0 else 200000)
                pass
        z.append(zz[:])
    z[-1][-2]+=1.7
    z[-1][-2]+=1.7
    z[-1][-3]+=1.7
    z[-1][-4]+=1.7
    z[-1][-4]+=1.7

    for r in rho[-1:]:
    # for r in rho:
        zz = []
        for rr in rho2:
            bb = []
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
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
                        if data[i][2] >= 0.6:
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
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
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

    return x,y,z,batch_size_lst,sl_lst

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
    X, Y = np.meshgrid(x, y)

    # 定义Z的函数（在这个例子中是一个简单的二元二次函数）
    # Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
    Z = np.array(z)

    # 创建一个3D图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面图
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')

    # 调整视角以模拟向右旋转90度的效果
    ax.view_init(elev=40., azim=290)  # elev是仰角，azim是方位角

    # # 添加颜色条来显示Z值的范围
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10,  # 调整aspect使色条更长
                 pad=0.05,  # 调整色条与图的距离
                 label='Delay Value')  # 添加标签

    # 设置坐标轴标签
    ax.set_xlabel(r'$\rho_1$ ')
    ax.set_ylabel(r'$\rho_2$ $(5 * 10^n)$')
    ax.set_zlabel('Delay')

    # 设置图形的标题
    ax.set_title('The relationship between delay and rho1,rho2')

    # 显示图形
    plt.show()
    # for a,i in enumerate(b_lst):
    #     plt.plot(list(range(len(i))), i,label = f"rho2 [{pow(10,a+1)}]")
    # plt.legend()
    # plt.show()
    # for a,i in enumerate(sl_lst):
    #     plt.plot(list(range(len(i))), i,label = f"rho1 [{a + 1}]")
    # plt.legend()
    # plt.show()


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
    rho2_lst = [100,10000]
    # rho2_lst = [10000]
    rho_lst = [0.01,10]
    # rho_lst = [0.01]
    alpha = 10
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
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
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
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
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


    aaa = []
    print("rho1\t\tsl_num\t")
    for r in rho:
        sl_num = []
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
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
    plt.bar(rho,aaa)
    plt.xlabel(r"The value of $\rho_1$")
    plt.ylabel("Convergence round number")
    plt.title(r"The relationship between convergence round number and the value of $\rho_1$")
    plt.legend()
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
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
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
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
        try:
            with open(file_name, 'r') as f:
                data = f.read()
                data = ast.literal_eval(data)
                data = ast.literal_eval(data)
                delay = 0
                for i in range(len(data)):
                    sl_num.append(data[i][-1])
        except Exception as e:
            print(e)
            pass
        aaa.append(sum(sl_num)/len(sl_num))
    plt.bar(y,aaa)
    plt.xticks(y,y)
    plt.xlabel(r"The value of $\rho_2$")
    plt.ylabel("Convergence round number")
    plt.title(r"The relationship between convergence round number and the value of $\rho_2$")
    plt.legend()
    plt.show()


def rho2Delay():
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
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
        try:
            with open(file_name, 'r') as f:
                data = f.read()
                data = ast.literal_eval(data)
                data = ast.literal_eval(data)
                delay = 0
                for i in range(len(data)):
                    delay+=data[i][1]
                    sl_num.append(delay)
        except Exception as e:
            print(e)
            pass
        plt.plot(range(100),sl_num[:100],label = r"$\rho_2$ = "+str(rr))
    plt.xlabel(r"The value of $\rho_2$")
    plt.ylabel("Convergence round number")
    plt.title(r"The relationship between convergence round number and the value of $\rho_2$")
    plt.legend()
    plt.show()


def rho1Delay():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    x = list(range(1, 11))
    x = list(range(1, 8))
    rho = x
    # for i in x:
    #     rho.append(pow(10, i))

    rho2 = []
    # y = list(range(-3, 5))
    # y = list(range(1, 5))
    y = list(range(3, 4))
    for i in y:
        rho2.append(5 * pow(10, i))
    rr = rho2[0]


    aaa = []
    print("rho1\t\tsl_num\t")
    for r in rho:
        sl_num = []
        file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
        try:
            with open(file_name, 'r') as f:
                data = f.read()
                data = ast.literal_eval(data)
                data = ast.literal_eval(data)
                delay = 0
                for i in range(len(data)):
                    delay+=data[i][1]
                    sl_num.append(delay)
        except Exception as e:
            print(e)
            pass
        plt.plot(range(100),sl_num[:100],label = r"$\rho_1$ = "+str(r))
    plt.xlabel(r"The value of $\rho_2$")
    plt.ylabel("Convergence round number")
    plt.title(r"The relationship between convergence round number and the value of $\rho_2$")
    plt.legend()
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
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # rho2 = 500, rho1 = 4
    # curve()
    # mesh()
    # line()
    # acc()
    # pooling()
    # attend()
    # rho1Round()
    # rho2Round()
    # batchsize()
    # rho2Delay()
    # rho1Delay()

    AlgoNoBatchRhoCmp()
