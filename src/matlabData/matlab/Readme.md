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
```angular2html
   'rho1:3,rho2:50'  ->  'rho1_3_rho2_50'
   'rho1:5,rho2:50'  ->  'rho1_5_rho2_50'
   'rho1:5,rho2:500'  ->  'rho1_5_rho2_500'
   'rho1:5,rho2:5000'  ->  'rho1_5_rho2_5000'
   'rho1:7,rho2:5000'  ->  'rho1_7_rho2_5000'
```

# subgradient.mat
- 本文件中
  - label 就是第几次坐标轮训
  - 序列值就是每次次梯度下降迭代中的目标函数值
  
# gibbs.mat
- 本文件中
  - label 就是第几次坐标轮训
  - 序列值就是每次gibbs算法迭代中的目标函数值
```angular2html
   'coordinate round 0'  ->  'coordinate_round_0'
   'coordinate round 1'  ->  'coordinate_round_1'
   'coordinate round 2'  ->  'coordinate_round_2'
   'coordinate round 3'  ->  'coordinate_round_3'
   'coordinate round 4'  ->  'coordinate_round_4'
```


# LearningPerformance.mat
- 本文件中
  - label 就是当前是accVsRound还是accVsDelay以及是横轴的round数目/delay 还是纵轴的test acc
  - 序列值就是test acc
```angular2html
   'type:Test accuracy vs round with non-iid param[1],mode:SL,value:round'  ->  'type_Test_accuracy_vs_round_with_non_iid_param_1__mode_SL_value_round'
   'type:Test accuracy vs round with non-iid param[1],mode:SL,value:acc'  ->  'type_Test_accuracy_vs_round_with_non_iid_param_1__mode_SL_value_acc'
   'type:Test accuracy vs round with non-iid param[1],mode:FL,value:round'  ->  'type_Test_accuracy_vs_round_with_non_iid_param_1__mode_FL_value_round'
   'type:Test accuracy vs round with non-iid param[1],mode:FL,value:acc'  ->  'type_Test_accuracy_vs_round_with_non_iid_param_1__mode_FL_value_acc'
   'type:Test accuracy vs round with non-iid param[1],mode:CHSFL,value:round'  ->  'type_Test_accuracy_vs_round_with_non_iid_param_1__mode_CHSFL_value_round'
   'type:Test accuracy vs round with non-iid param[1],mode:CHSFL,value:acc'  ->  'type_Test_accuracy_vs_round_with_non_iid_param_1__mode_CHSFL_value_acc'
   'type:Test accuracy vs round with non-iid param[1],mode:AlgoOnlyBatch,value:round'  ->  'type_Test_accuracy_vs_round_with_non_iid_param_1__mode_AlgoOnlyBatch_value_round'
   'type:Test accuracy vs round with non-iid param[1],mode:AlgoOnlyBatch,value:acc'  ->  'type_Test_accuracy_vs_round_with_non_iid_param_1__mode_AlgoOnlyBatch_value_acc'
   'type:Test accuracy vs round with non-iid param[1],mode:AlgoWithBatch,value:round'  ->  'type_Test_accuracy_vs_round_with_non_iid_param_1__mode_AlgoWithBatch_value_round'
   'type:Test accuracy vs round with non-iid param[1],mode:AlgoWithBatch,value:acc'  ->  'type_Test_accuracy_vs_round_with_non_iid_param_1__mode_AlgoWithBatch_value_acc'
   'type:Test accuracy vs round with non-iid param[1],mode:HSFLAlgo,value:round'  ->  'type_Test_accuracy_vs_round_with_non_iid_param_1__mode_HSFLAlgo_value_round'
   'type:Test accuracy vs round with non-iid param[1],mode:HSFLAlgo,value:acc'  ->  'type_Test_accuracy_vs_round_with_non_iid_param_1__mode_HSFLAlgo_value_acc'
```

# batchRoundAlgo.mat
- 其中label 就是 当前的 ut 值是 before floor/ after floor / after_round ,之后的rho1是y轴,rho2是x轴
- 值是当前组合和状态下的ut值
```angular2html
'before_rho1_500,rho2_3'  ->  'before_rho1_500_rho2_3'
   'floor_rho1_500,rho2_3'  ->  'floor_rho1_500_rho2_3'
   'after_rho1_500,rho2_3'  ->  'after_rho1_500_rho2_3'
   'before_rho1_2000,rho2_3'  ->  'before_rho1_2000_rho2_3'
   'floor_rho1_2000,rho2_3'  ->  'floor_rho1_2000_rho2_3'
   'after_rho1_2000,rho2_3'  ->  'after_rho1_2000_rho2_3'
   'before_rho1_5000,rho2_3'  ->  'before_rho1_5000_rho2_3'
   'floor_rho1_5000,rho2_3'  ->  'floor_rho1_5000_rho2_3'
   'after_rho1_5000,rho2_3'  ->  'after_rho1_5000_rho2_3'
   'before_rho1_500,rho2_4'  ->  'before_rho1_500_rho2_4'
   'floor_rho1_500,rho2_4'  ->  'floor_rho1_500_rho2_4'
   'after_rho1_500,rho2_4'  ->  'after_rho1_500_rho2_4'
   'before_rho1_2000,rho2_4'  ->  'before_rho1_2000_rho2_4'
   'floor_rho1_2000,rho2_4'  ->  'floor_rho1_2000_rho2_4'
   'after_rho1_2000,rho2_4'  ->  'after_rho1_2000_rho2_4'
   'before_rho1_5000,rho2_4'  ->  'before_rho1_5000_rho2_4'
   'floor_rho1_5000,rho2_4'  ->  'floor_rho1_5000_rho2_4'
   'after_rho1_5000,rho2_4'  ->  'after_rho1_5000_rho2_4'
   'before_rho1_500,rho2_5'  ->  'before_rho1_500_rho2_5'
   'floor_rho1_500,rho2_5'  ->  'floor_rho1_500_rho2_5'
   'after_rho1_500,rho2_5'  ->  'after_rho1_500_rho2_5'
   'before_rho1_2000,rho2_5'  ->  'before_rho1_2000_rho2_5'
   'floor_rho1_2000,rho2_5'  ->  'floor_rho1_2000_rho2_5'
   'after_rho1_2000,rho2_5'  ->  'after_rho1_2000_rho2_5'
   'before_rho1_5000,rho2_5'  ->  'before_rho1_5000_rho2_5'
   'floor_rho1_5000,rho2_5'  ->  'floor_rho1_5000_rho2_5'
   'after_rho1_5000,rho2_5'  ->  'after_rho1_5000_rho2_5'
   'before_rho1_500,rho2_6'  ->  'before_rho1_500_rho2_6'
   'floor_rho1_500,rho2_6'  ->  'floor_rho1_500_rho2_6'
   'after_rho1_500,rho2_6'  ->  'after_rho1_500_rho2_6'
   'before_rho1_2000,rho2_6'  ->  'before_rho1_2000_rho2_6'
   'floor_rho1_2000,rho2_6'  ->  'floor_rho1_2000_rho2_6'
   'after_rho1_2000,rho2_6'  ->  'after_rho1_2000_rho2_6'
   'before_rho1_5000,rho2_6'  ->  'before_rho1_5000_rho2_6'
   'floor_rho1_5000,rho2_6'  ->  'floor_rho1_5000_rho2_6'
   'after_rho1_5000,rho2_6'  ->  'after_rho1_5000_rho2_6'
```


# iid_cmp.mat
- 其中 label 就是 当前的柱状图是处于哪一个准确度,以及当前准确度下非独立同分布系数alpha是几
- 值是当前准确度和非独立同分布系数下达到目标准确度所需要的时延值
```angular2html
   'acc=0.55&$\\alpha$=0.1'  ->  'acc_0_55___alpha__0_1'
   'acc=0.55&$\\alpha$=1'  ->  'acc_0_55___alpha__1'
   'acc=0.55&$\\alpha$=10'  ->  'acc_0_55___alpha__10'
   'acc=0.5&$\\alpha$=1'  ->  'acc_0_5___alpha__1'
   'acc=0.5&$\\alpha$=10'  ->  'acc_0_5___alpha__10'
```


