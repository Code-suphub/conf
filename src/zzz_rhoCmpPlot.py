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


def curve():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    x = list(range(-4, 9))
    for i in x:
        rho.append(pow(10, i))

    rho2 = []
    # y = list(range(-3, 5))
    y = list(range(2, 5))
    # y = list(range(3, 4))
    for i in y:
        rho2.append(pow(10, i))

    z = []

    for rr in rho2:
    # for r in rho:
        zz = []
        for r in rho:
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
            try:
                with open(file_name, 'r') as f:
                    data = f.read().split("\n")
                    delay = 0
                    for i in range(len(data)):
                        data[i] = [float(j) for j in data[i].split(",")]
                        delay += data[i][1]*0.0001
                        if data[i][2] >= 0.5:
                            zz.append(delay)
                            break
            except Exception as e:
                zz.append(max(zz) if len(zz) > 0 else 200000)
                pass
        z.append(zz[:])

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

    # 添加颜色条来显示Z值的范围
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # 设置坐标轴标签
    ax.set_xlabel('rho1')
    ax.set_ylabel('rho2')
    ax.set_zlabel('delay')

    # 设置图形的标题
    ax.set_title('The relationship between delay and rho1,rho2')

    # 显示图形
    plt.show()




def mesh():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rho = []  # X
    x = list(range(-4, 9))
    for i in x:
        rho.append(pow(10, i))

    rho2 = []
    # y = list(range(-3, 5))
    y = list(range(2, 5))
    # y = list(range(3, 4))
    for i in y:
        rho2.append(pow(10, i))

    z = []

    for rr in rho2:
    # for r in rho:
        zz = []
        for r in rho:
            file_name = f"../save/output/conference/cmpResult/rho/local_cnt[1]_user30_rho1[{r}]_rho2[{rr}]_alpha[10].csv"
            try:
                with open(file_name, 'r') as f:
                    data = f.read().split("\n")
                    delay = 0
                    for i in range(len(data)):
                        data[i] = [float(j) for j in data[i].split(",")]
                        delay += data[i][1]*0.0001
                        if data[i][2] >= 0.5:
                            zz.append(delay)
                            break
            except Exception as e:
                zz.append(max(zz) if len(zz) > 0 else 200000)
                pass
        z.append(zz[:])

    # 生成X和Y的数据网格
    x = np.array(x)
    y = np.array(y)
    # x = np.linspace(-5, 5, 100)
    # y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    # 定义Z的函数（在这个例子中是一个简单的二元二次函数）
    # Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
    Z = np.array(z)

    # 绘制3D网格图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=100, cstride=100, alpha=0.5)  # 绘制网格框架，不填充颜色

    # 在每个(X, Y)网格点上显示z值
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(X[i, j], Y[i, j]+pow(-1,j)*0.2, Z[i, j], f'{Z[i, j]:.4f}', ha='center', va='bottom')

    # 设置轴的范围和标签
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(0, np.max(Z) * 1.1)  # 稍微增加z轴的上限以更好地显示数据
    # 设置坐标轴标签
    ax.set_xlabel('rho1')
    ax.set_ylabel('rho2')
    ax.set_zlabel('delay')

    # 设置图形的标题
    ax.set_title('The relationship between delay and rho1,rho2')

    # 显示图表
    plt.show()

if __name__ == '__main__':
    # curve()
    # mesh()
    # line()
    acc()