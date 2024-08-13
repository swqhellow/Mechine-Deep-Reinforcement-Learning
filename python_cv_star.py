import random
import cv2
import numpy as np
import os
import time
import math
RED = 1
GREEN = 2
BLUE = 3
BLACK = 4
WHITE = 5
GRAY = 6
YELLOW = 7
CYAN = 8
MAGENTA = 9
COLORS = [
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [0, 0, 0],
    [255, 255, 255],
    [128, 128, 128],
    [0, 255, 255],
    [255, 255, 0],
    [255, 0, 255]
]
# while True:
#     img = cv2.imread(r"downloaded_images\tree\image_5.jpg")
#     cv2.rectangle(img, (i*speed, i*speed), (i*speed+10, i*speed+10), WHITE, -1)
#     cv2.rectangle(img, (i, i), (i+10, i+10), WHITE, -1)
#     cv2.imshow('gif', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # 显示每张图像 500 毫秒，按 'q' 键退出
#         break
#     i+=1
#     if(i>=190):i=0
# cv2.destroyAllWindows()
# imgsfolder = 'downloaded_images/computer'
# imgs_path = os.listdir(imgsfolder)
# print(imgs_path)
# for img_path in imgs_path:
#     img_path = f'{imgsfolder}/{img_path}'
#     if not os.path.exists(img_path):
#         print(f"File not found: {img_path}")
#     else:
#         image = cv2.imread(img_path)
#         image = cv2.resize(image,(640,480))
#     cv2.imshow('img',image)
#     if cv2.waitKey(-1) & 0xff == ord('q'):
#         continue
# 创建一个空白图像

def draw_heart(img, center, size, color):
    # 中心点和大小
    x, y = center
    width, height = size

    # 左半圆
    left_circle_center = (x - width // 4, y - height // 4)
    left_circle_radius = width // 4

    # 右半圆
    right_circle_center = (x + width // 4, y - height // 4)
    right_circle_radius = width // 4

    # 下半部分的多边形顶点
    points = np.array([
        [x - width // 2, y - height // 4],
        [x + width // 2, y - height // 4],
        [x, y + height // 2]
    ], np.int32)
    points = points.reshape((-1, 1, 2))

    # 绘制左半圆
    cv2.circle(img, left_circle_center, left_circle_radius, color, -1)

    # 绘制右半圆
    cv2.circle(img, right_circle_center, right_circle_radius, color, -1)

    # 绘制下半部分的多边形
    cv2.fillPoly(img, [points], color)

def drawstar(image, center, radius, color, rotate_angle=0):
    # 定义两个半径
    outer_radius = radius
    inner_radius = radius / 2.5

    # 计算五角星的顶点（外顶点和内顶点交替排列）
    points = []
    for i in range(10):
        angle = (i * 36 * math.pi / 180) + rotate_angle  # 每36度一个顶点（外顶点和内顶点交替排列）
        if i % 2 == 0:
            r = outer_radius
        else:
            r = inner_radius
        x = int(center[0] + r * math.cos(angle))
        y = int(center[1] - r * math.sin(angle))
        points.append((x, y))

    # 将顶点列表转换为 NumPy 数组
    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))

    # 使用 cv2.fillPoly 函数绘制实心五角星
    cv2.fillPoly(image, [points], color=color)
# 定义流星的属性
width, height = 800, 600
image = np.zeros((height, width, 3), dtype='uint8')
num_meteors = 20
meteors = []
star_size =10
line_thickness = 6
for _ in range(num_meteors):
    x = random.randint(0, width)
    y = random.randint(0, height)
    speed = random.randint(2, 10)
    length = random.randint(50, 100)
    randcolorint = (np.random.randint(1, 255, [3]))
    meteors.append([x, y, speed, length, randcolorint.tolist()])

# 设置窗口
cv2.namedWindow('Meteor Shower', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Meteor Shower', width, height)
heart_img = cv2.imread(r'Learning\heart.jpg')
heart_img = cv2.resize(heart_img,(400,400))
print(heart_img.shape)
print(image.shape)
image[100:500,200:600,] = heart_img
# 生成流星效果
while True:
    image[100:500,200:600,] = heart_img
    # 减少图像亮度，产生拖影效果
    # image = np.zeros((height, width, 3), dtype='uint8')
    #draw_heart(image,(int(width/4),int(height/2)),[100,100],[10,10,255])
    cv2.putText(image,"SWQ Love GY",(int(width/2.6),int(height/2)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,[255,0,0],2)
    for meteor in meteors:
        x, y, speed, length, color = meteor
        x = x-speed
        y = y-speed
        # 绘制流星
        # cv2.rectangle(image, (x + length, y + length), (x + length+10, y + length+10), meteor[4], 4)
        drawstar(image, (x + length, y + length), star_size, [0,0,0], x % 144)
        cv2.line(image, (x, y), (x + length, y + length), meteor[4], line_thickness)

    image = cv2.addWeighted(image, 0.9, np.zeros_like(image), 0.1, 0)

    for meteor in meteors:
        
        x, y, speed, length, color = meteor
        # 绘制流星
        cv2.line(image, (x, y), (x + length, y + length), meteor[4], line_thickness)
        # cv2.rectangle(image, (x + length, y + length), (x + length+10, y + length+10), meteor[4], 4)
        drawstar(image, (x + length, y + length), star_size, meteor[4], x % 144)
        # 更新流星位置
        meteor[0] += speed
        meteor[1] += speed
        
        # 如果流星移出屏幕，重置它的位置
        if meteor[0] > width or meteor[1] > height:
            meteor[0] = random.randint(0, width)
            meteor[1] = random.randint(0, height)
            meteor[2] = random.randint(5, 15)
            meteor[3] = random.randint(50, 100)

    # 显示图像
    cv2.imshow('Meteor Shower', image)

    # 按下 'q' 键退出
    if (cv2.waitKey(16) & 0xFF) == ord('q'):
        break
cv2.destroyAllWindows()
