# curve.mat
- 本文件中
  - rho2就是x轴，表示[50,200,500,2000,5000,20000,50000] 的序列值
  - rho1就是y轴，表示[3,4,5,6,7,8,9] 的序列值 
  - delay：纵轴,表示当前rho1 , rho2下收敛需要的时延

# ut_value.mat
- 本文件中
  - rho2就是x轴，表示[50,200,500,2000,5000,20000,50000] 的序列值
  - rho1就是y轴，表示[3,4,5,6,7,8,9] 的序列值 
  - ut_value：纵轴,表示当前rho1 , rho2下坐标轮训收敛时的目标函数值



# batchsize.mat
- 本文件中
  - rho2就是横轴，表示[50,200,500,2000,5000,20000,50000] 的序列值
  - Batchsize 就是第一个纵轴值，表示当前rho2下所有设备的批次大小总量 
  - round_num_lst： 表示当前rho2下收敛需要的轮次

# attend.mat
- 本文件中
  - rho1就是横轴，表示[3,4,5,6,7,8,9] 的序列值
  - SL_device_num 就是第一个纵轴值，表示当前rho1下所有SL设备数量 
  - round_num_lst： 表示当前rho1下收敛需要的轮次


# rho2VsDelay.mat
- 本文件中
  - rho2就是横轴，表示[50,200,500,2000,5000,20000,50000] 的序列值
  - delay 就是第纵轴值，表示当前rho2下收敛需要的时延值

# rho1VsDelay.mat
- 本文件中
  - rho就是横轴，表示[3,4,5,6,7,8,9] 的序列值
  - delay 就是第纵轴值，表示当前rho1下收敛需要的时延值


# coordinate.mat
- 本文件中
  - label 就是(rho1,rho2) 的组合
  - 序列值就是迭代中每个周期的目标函数值

# subgradient.mat
- 本文件中
  - label 就是第几次坐标轮训
  - 序列值就是每次次梯度下降迭代中的目标函数值
  
# gibbs.mat
- 本文件中
  - label 就是第几次坐标轮训
  - 序列值就是每次gibbs算法迭代中的目标函数值


# LearningPerformance.mat
- 本文件中
  - label 就是当前是accVsRound还是accVsDelay以及是横轴的round数目/delay 还是纵轴的test acc
  - 序列值就是test acc

# batchRoundAlgo.mat
- 其中label 就是 当前的 ut 值是 before floor/ after floor / after_round ,之后的rho1是y轴,rho2是x轴
- 值是当前组合和状态下的ut值


# iid.mat
- 其中 label 就是 当前的柱状图是处于哪一个准确度,以及当前准确度下非独立同分布系数alpha是几
- 值是当前准确度和非独立同分布系数下达到目标准确度所需要的时延值


