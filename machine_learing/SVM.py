import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 数据
X = np.array([[2, 3], [3, 4], [4, 5], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, -1, -1, -1])

# 创建SVM模型，使用RBF核
clf = svm.SVC(kernel='linear', gamma='scale', C=1.0)

# 训练模型
clf.fit(X, y)

# sdc

# 创建网格来绘制决策边界
xx, yy = np.meshgrid(np.linspace(0, 5, 500), np.linspace(0, 6, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制数据点
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')

# 绘制决策边界和支持向量
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[
            :, 1], s=100, facecolors='none', edgecolors='black', label='Support Vectors')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM with RBF Kernel')
plt.legend()
plt.show()
