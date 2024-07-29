# -*- coding :UTF-8 -*-
"""
@author：shangwenqing
@file:logistic_regression.py
@time:2024:07:29:9:55
@IDE:PyCharm
@copyright:WenQing Shang
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 生成示例数据
np.random.seed(0)
X = np.random.rand(100, 2) * 10 - 5  # 两个特征的样本点
# Y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 标签：如果两个特征之和大于0，则为1，否则为0
Y = np.random.choice([1, 0], 100, p=[0.5, 0.5])
# 拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 创建逻辑回归模型并进行拟合
model = LogisticRegression()
model.fit(X_train, Y_train)

# 预测
Y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# 可视化决策边界
xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', marker='o', s=100, alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
