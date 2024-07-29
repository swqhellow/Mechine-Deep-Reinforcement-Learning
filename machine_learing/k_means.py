# -*- coding :UTF-8 -*-
"""
@作者：shangwenqing
@文件名:k_means.py
@时间:2024:07:23:10:51
@IDE:PyCharm
@copyright:WenQing Shang
"""
import numpy as np
import matplotlib.pyplot as plt


def initialize_centroids(X, k):
    # 随机选4个择样本点作为初始化的点
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]


def assign_clusters(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)


def update_centroids(X, labels, k):
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids


def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        old_centroids = centroids
        labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, k)

        if np.all(np.abs(centroids - old_centroids) < tol):
            break
    return centroids, labels


# 示例数据
#X = np.array([[1, 2], [1, 4], [1, 0],
              #[10, 2], [10, 4], [10, 0]])

# 运行k-means算法



# 可视化结果
def plot_kmeans(X, centroids, labels, k):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    plt.figure(figsize=(8, 6))

    for i in range(k):
        points = X[labels == i]
        plt.scatter(points[:, 0], points[:, 1], s=100, c=colors[i], label=f'Cluster {i}')

    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='x', label='Centroids')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('K-means Clustering')
    plt.show()
np.random.seed(22)
X = np.random.uniform(0,1000,(100,2))
# 设置簇数
k = 4
centroids, labels = kmeans(X, k)
plot_kmeans(X, centroids, labels, k)
