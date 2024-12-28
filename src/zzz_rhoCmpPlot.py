from matplotlib import pyplot as plt

import common
rho_lst = common.rho_lst
slLst = []
delayLst = []
delayLstAvg = []
totalRound = []
for rho in rho_lst:
    try:
        with open("../save/output/conference/cmpResult/rho/cnt[1]_user30_" + str(rho) + ".csv", 'r') as f:
            res=  [i.split(",") for i in f.read().split("\n")][:-1]
            slCnt =[]
            delaySum = []
            for d in res:
                if float(d[2]) <0.6:
                    slCnt.append(int(d[0]))
                    delaySum.append(float(d[1]))
            slLst.append(sum(slCnt)/len(slCnt))
            delayLst.append(sum(delaySum))
            delayLstAvg.append(sum(delaySum)/len(delaySum))
            totalRound.append(len(delaySum))
    except:
        pass

rho_lst = common.rho_lst[:len(totalRound)]
plt.plot([r for r in rho_lst],slLst,label = "SL Number Per Round")
# plt.plot([1/r for r in rho_lst],delayLst,label = "Overall Latency")
plt.plot([r for r in rho_lst],delayLstAvg,label = "Latency Per Round")
plt.plot([r for r in rho_lst],totalRound,label = "Total number of Round")
#         data.append()
#     plt.plot([i for i in range(len(data[-1]))],data[-1],label = f"{sigma}")
#
plt.legend()
plt.show()