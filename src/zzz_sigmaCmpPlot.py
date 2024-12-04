from matplotlib import pyplot as plt

import common

def innerCmpPlot():
    data = []
    for sigma in common.sigma_lst:
        with open("../save/output/conference/cmpResult/sigma/inner/sigma_" + str(sigma) + ".csv", 'r') as f:
            data.append([float(i) for i in f.read().split(",")[1:]])
        plt.plot([i for i in range(len(data[-1]))], data[-1], label=f"sigma={sigma}")
    plt.xlabel("iteration round")
    plt.ylabel("target function u_t")

    plt.legend()
    plt.show()


rho2_lst = [10,50,100,500,1000,5000,10000]
def outterCmpPlot():
    data = []
    for rho2 in rho2_lst:
        try:
            with open("../save/output/conference/cmpResult/sigma/outter/sigma_" + str(0.001) + f"_rho2_{rho2}.csv", 'r') as f:
                data.append([eval(i) for i in f.read().split("---")[1:]])
            plt.plot([i for i in range(len(data[-1]))], [i[0] for i in data[-1]], label=f"rho2={rho2}")
        except Exception as e:
            pass
    plt.xlabel("iteration round")
    plt.ylabel("target function u_t")

    plt.legend()
    plt.show()

    data = []
    for rho2 in rho2_lst:
        try:
            with open("../save/output/conference/cmpResult/sigma/outter/sigma_" + str(0.001) + f"_rho2_{rho2}.csv",
                      'r') as f:
                data.append([eval(i) for i in f.read().split("---")[1:]])
            plt.plot([i for i in range(len(data[-1]))], [i[1] for i in data[-1]], label=f"rho2={rho2}")
        except Exception as e:
            pass
    plt.xlabel("iteration round")
    plt.ylabel("batch-size")

    plt.legend()
    plt.show()

rho2_lst = [0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000]
def outterCmpPlot2():
    # path_pre = "../save/output/conference/cmpResult/sigma/outter/optimal_sigma_"
    path_pre = "../save/output/conference/cmpResult/sigma/outter/new_sigma_"

    data = []
    for rho2 in rho2_lst:
        try:
            with open(path_pre + str(0.001) + f"_rho2_{rho2}.csv", 'r') as f:
                data.append([eval(i) for i in f.read().split("---")[1:]])
            plt.plot([i for i in range(len(data[-1]))], [i[0] for i in data[-1]], label=f"rho2={rho2}")
        except Exception as e:
            pass
    plt.xlabel("iteration round")
    plt.ylabel("target function u_t")
    plt.title("target function vs iteration round")

    plt.legend()
    plt.show()

    data = []
    for rho2 in rho2_lst:
        try:
            with open(path_pre + str(0.001) + f"_rho2_{rho2}.csv",
                      'r') as f:
                data.append([eval(i) for i in f.read().split("---")[1:]])
            plt.plot([i for i in range(len(data[-1]))], [i[1] for i in data[-1]], label=f"rho2={rho2}")
        except Exception as e:
            pass
    plt.xlabel("iteration round")
    plt.ylabel("batch-size")
    plt.title("batch-size vs iteration round")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # outterCmpPlot()
    outterCmpPlot2()