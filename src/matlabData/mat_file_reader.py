# import scipy.io
#
# data = scipy.io.loadmat('matlab/coordinate.mat')
#
# # 查看变量名
# print(data.keys())
#
# # 访问变量
# matrix = data['my_matrix']
# print(matrix.shape)

import numpy as np
from scipy.io import loadmat

# 1. 读文件
data = loadmat('matlab/coordinate.mat')         # 返回一个 dict
data = loadmat('matlab/subgradient.mat')         # 返回一个 dict
data = loadmat('matlab/gibbs.mat')         # 返回一个 dict
# data = loadmat('matlab/rho2VsDelay.mat')         # 返回一个 dict
# data = loadmat('matlab/LearningPerformance_delay.mat')         # 返回一个 dict
data = loadmat('curve.mat')         # 返回一个 dict
data = loadmat('LearningPerformance_delay.mat')         # 返回一个 dict
# data = loadmat('rho2VsDelay.mat')         # 返回一个 dict


vars_only = {k: v for k, v in data.items() if not k.startswith('__')}
for name, arr in vars_only.items():
    if isinstance(arr, np.ndarray) and arr.ndim == 2:
        print(f'\n{name} ({arr.shape}):')
        print(arr)

# # 2. 取矩阵变量
# # 假设文件里只有一个真正的矩阵变量 'A'（忽略 __header/__version/__globals）
# A = data['A']          # shape 如 (m, n)
#
# # 3. 按矩阵打印
# np.set_printoptions(precision=4, suppress=True)   # 可选：控制小数位
# print(A)