import cv2

# 创建一个窗口，用于显示摄像头的图像
cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

# 使用cv2.VideoCapture(0)打开摄像头，0是默认的摄像头设备索引
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 循环读取摄像头的每一帧
while True:
    # 读取一帧
    ret, frame = cap.read()

    # 如果正确读取帧，ret为True
    if not ret:
        print("无法读取帧")
        break

    # 显示帧
    cv2.imshow('Camera', frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) == ord('q'):
        break
# 释放摄像头
cap.release()
# 销毁所有窗口
cv2.destroyAllWindows()
