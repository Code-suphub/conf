# import matplotlib.pyplot as plt
# fontSize = 18
# # fontSize = 32
# plt.rcParams.update({
#     'axes.labelsize': fontSize,  # X/Y轴标签字体大小
#     'axes.titlesize': fontSize,  # 标题字体大小
#     'legend.fontsize': fontSize,  # 图例字体大小
#     'xtick.labelsize': fontSize,  # X轴刻度标签字体
#     'ytick.labelsize': fontSize  # Y轴刻度标签字体
# })
#
# plt.rcParams['font.sans-serif'] = ['AR PL UMing CN']  # 中文字体
# plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
# plt.rcParams['figure.figsize'] = (8, 4)
#
# y = [30 - i * (i - 1) / 30 for i in range(31)]
#
# plt.plot(range(31), y)
#
# # 隐藏x轴和y轴的刻度线
# plt.gca().tick_params(axis='y', which='both', left=False, labelleft=False)  # 隐藏y轴刻度线及标签
# plt.gca().tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # 隐藏x轴刻度线及标签
#
# plt.xlabel("SL 设备数量")
# plt.ylabel("损失函数差值上界")
# plt.show()


import matplotlib.pyplot as plt
fontSize = 18
# fontSize = 32
plt.rcParams.update({
    'axes.labelsize': fontSize,  # X/Y轴标签字体大小
    'axes.titlesize': fontSize,  # 标题字体大小
    'legend.fontsize': fontSize,  # 图例字体大小
    'xtick.labelsize': fontSize,  # X轴刻度标签字体
    'ytick.labelsize': fontSize  # Y轴刻度标签字体
})

plt.rcParams['font.sans-serif'] = ['AR PL UMing CN']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
plt.rcParams['figure.figsize'] = (8, 4)

y = [10000/i for i in range(1,2000)]

plt.plot(range(1,2000), y)

# 隐藏x轴和y轴的刻度线
plt.gca().tick_params(axis='y', which='both', left=False, labelleft=False)  # 隐藏y轴刻度线及标签
plt.gca().tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # 隐藏x轴刻度线及标签

plt.xlabel("设备批次大小")
plt.ylabel("损失函数差值上界")
plt.show()