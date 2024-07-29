# -*- coding :UTF-8 -*-
"""
@author：shangwenqing
@file:A_Start.py
@time:2024:07:28:19:39
@IDE:PyCharm
@copyright:WenQing Shang
"""
import heapq
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # 实际代价
        self.h = 0  # 启发式代价
        self.f = 0  # 总代价

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    # 使用曼哈顿距离作为启发式函数
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, end):
    # 创建起始节点和目标节点
    start_node = Node(start)
    end_node = Node(end)

    # 初始化开放列表和封闭列表
    open_list = []
    closed_list = set()

    # 将起始节点添加到开放列表
    heapq.heappush(open_list, start_node)

    # 定义4个可能的移动方向（上下左右）
    move_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    path = []
    while open_list:
        # 获取开放列表中具有最低总代价的节点
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        # 如果当前节点是目标节点，回溯路径
        if current_node.position == end_node.position:
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        # 生成当前节点的邻居
        for move in move_directions:
            neighbor_position = (current_node.position[0] + move[0], current_node.position[1] + move[1])

            # 检查邻居位置是否在网格范围内
            if 0 <= neighbor_position[0] < len(grid) and 0 <= neighbor_position[1] < len(grid[0]):
                # 检查邻居是否为障碍物
                if grid[neighbor_position[0]][neighbor_position[1]] != 0:
                    continue

                # 创建邻居节点
                neighbor_node = Node(neighbor_position, current_node)

                # 如果邻居节点在封闭列表中，跳过
                if neighbor_node.position in closed_list:
                    continue

                # 计算邻居节点的代价
                neighbor_node.g = current_node.g + 1
                neighbor_node.h = heuristic(neighbor_node.position, end_node.position)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                # 如果邻居节点在开放列表中，且新的代价更低，更新代价
                if any(neighbor.position == neighbor_node.position and neighbor.f <= neighbor_node.f for neighbor in open_list):
                    continue

                # 将邻居节点添加到开放列表
                heapq.heappush(open_list, neighbor_node)

                # 可视化每一步的过程
                path.append(current_node.position)
                yield path

    yield None  # 无路径可达

rows = 100
cols = 100
p = 0.1  # 1出现的概率
# 初始化示例网格（0表示可通行，1表示障碍物）
np.random.seed(10)
grid = np.random.choice([0, 1], size=(rows, cols), p=[1 - p, p])
# 保证起始点为0
grid[0][0] = 0
# ensure the ending is zero
grid[rows - 1][cols - 1] = 0

grid[40:60,40:80] = 1
# define the start and end
start = (0, 0)
end = (rows - 1, cols - 1)

# set the image
fig, ax = plt.subplots()
image = np.zeros((rows, cols, 3), dtype=np.uint8)
image[grid == 1] = [255, 0, 0]  # 红色代表障碍物red is the obstacle


image[start] = [0, 0, 255]  # 蓝色代表起点
image[end] = [255, 255, 0]  # 黄色代表终点
img = ax.imshow(image)

# 更新函数
def update(path):
    if path is not None:
        for pos in path:
            image[pos] = [0, 255, 0]  # 绿色代表路径
        img.set_data(image)
    return img,

# 动画
ani = FuncAnimation(fig, update, frames=astar(grid, start, end), repeat=False, blit=True,interval=1)
plt.show()
