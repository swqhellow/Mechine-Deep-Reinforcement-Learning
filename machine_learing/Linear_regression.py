# -*- coding :UTF-8 -*-
"""
@author：shangwenqing
@file:Linear_regression.py
@time:2024:07:29:9:32
@IDE:PyCharm
@copyright:WenQing Shang
"""
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)

# 添加偏置项，即常数项1
X_b = np.c_[np.ones((100, 1)), X]

# 正规方程求解回归系数
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)

# 打印回归系数
print(f"Intercept (beta_0): {theta_best[0][0]}")
print(f"Slope (beta_1): {theta_best[1][0]}")

# 使用回归系数进行预测
Y_pred = X_b.dot(theta_best)

# 可视化
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
