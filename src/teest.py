import cvxpy as cp
import numpy as np

# 假设 K 是一个已知的索引集，例如 K = {1, 2, 3}
xi = []
for i in range()

# 定义目标函数
a = 1  # 假设常数 a 为 1，你可以根据需要修改
objective = cp.Maximize(3 * x1 + 4*x2 )

constants = [
    x1+2*x2<=8,
    3*x1 + 2 *x2 <=12,
    x1>=0,
    x2>=0
]

# 定义并求解问题
problem = cp.Problem(objective,constants)
problem.solve()

# 输出结果
print("Optimal values of m_k:", x1.value)
print("Optimal values of m_k:", x2.value)
print("Optimal objective value:", problem.value)
