import tkinter as tk
from tkinter import Label, Button, Frame
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from tkinter import filedialog
import re
import os

# 定义视频和图片的文件扩展名
VIEDO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'mpeg', '3gp']
IMAGE_EXTENSIONS = ['jpg', 'png']

# 定义矩形框的颜色列表
COLORS = [
    [0, 0, 255],   # 红色
    [0, 255, 0],   # 绿色
    [255, 0, 0],   # 蓝色
    [0, 0, 0],     # 黑色
    [255, 255, 255],  # 白色
    [128, 128, 128],  # 灰色
    [0, 255, 255],    # 黄色
    [255, 255, 0],    # 青色
    [255, 0, 255]     # 品红色
]

# 加载 COCO 标签文件
def loadcocolabel(label_path):
    label = []
    with open(label_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 移除无关字符并获取标签名
            line = line.replace('"', '').replace(',', '').replace('\n', '').split('.')
            line = re.sub(r'\(.*?\)', '', line[1]).strip()  # 去掉括号内容
            label.append(line)
    return label

# 使用 YOLO 模型进行目标检测
def detecion(img, model, label):
    pre = model(img)
    boxes = pre[0].boxes
    classes = []
    for box, cls in zip(boxes.xyxy, boxes.cls):
        x1, y1, x2, y2 = map(int, box)  # 获取检测框的坐标
        cls = int(cls)  # 获取类别索引
        # 绘制检测框和标签
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[(cls % 9)], 2)
        cv2.putText(img, f'{label[cls]}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, [255, 0, 0], 2)
        classes.append(label[cls])  # 保存检测到的类别
    return img, classes

# 获取默认的视频文件
def get_defaultvideo():
    files = os.listdir()
    for file in files:
        ext = file.split('.', -1)[-1]
        if ext in VIEDO_EXTENSIONS:
            return file

# 视频播放类
class VideoPlayer:
    def __init__(self, root):
        # 加载 YOLO 模型和标签
        self.model = YOLO(r'Learning\deep_learning\yoloGUI\yolov8n.pt')
        self.labels = loadcocolabel(r'Learning\deep_learning\yoloGUI\lable.txt')

        # 设置根窗口属性
        self.root = root
        self.root.title("YOLO GUI")
        self.root.geometry("1000x600")

        # 创建视频显示框架
        self.video_frame = Frame(root)
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 创建用于显示视频帧的Label
        self.label = Label(self.video_frame)
        self.label.pack()

        # 创建文本显示框
        self.text_frame = tk.Text(root, height=10, width=40, font=("Arial", 12))
        self.text_frame.pack(padx=10)
        self.showtext("This is a detection GUI")  # 初始化文本框

        # 创建底部的控制按钮区域
        self.controls_frame = Frame(root)
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # 创建选择文件按钮
        self.selectfile = Button(self.controls_frame,text='select img or viedo',foreground='blue', command=self.get_viedopath)
        self.selectfile.pack(side=tk.LEFT, padx=10, pady=10)

        # 创建打开摄像头按钮
        self.opencamera = Button(self.controls_frame, text='open camera', command=self.start_camera)
        self.opencamera.pack(side=tk.LEFT, padx=10, pady=10)

        # 创建开始播放按钮
        self.start_button = Button(self.controls_frame, text="start", command=self.start_video)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

        # 创建暂停播放按钮
        self.pausebutton = Button(self.controls_frame, text="pause", command=self.pausevideo)
        self.pausebutton.pack(side=tk.LEFT, padx=10, pady=10)

        # 创建停止播放按钮
        self.stop_button = Button(self.controls_frame, text="end", command=self.stop_videoorcamera)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=10)

        # 创建检测按钮
        self.det_button = Button(self.controls_frame, text="detection", command=self.turn_detection)
        self.det_button.pack(side=tk.LEFT, padx=10, pady=10)

        # 初始化视频播放控制变量
        self.cap = None  # 视频捕捉对象
        self.playing = False  # 播放状态
        self.ispause = None  # 暂停状态
        self.detec = None  # 检测状态
        self.videofilepath = get_defaultvideo()  # 默认视频文件路径
        self.showtext(f'load default viedo:{self.videofilepath}')
        self.imgpath = None  # 图像文件路径
        self.iscamera = None  # 摄像头状态

    # 获取视频路径
    def get_viedopath(self):
        self.videofilepath = self.getfilepath()
        if self.videofilepath:
            ext = self.videofilepath.split('.', 1)[1]
            if ext in VIEDO_EXTENSIONS:
                self.showtext(f"succesed set viedo path {self.videofilepath}!")
            elif ext in IMAGE_EXTENSIONS:
                self.imgpath = self.videofilepath
                self.videofilepath = None
                self.showtext(f"succesed load {self.imgpath}!")
            else:
                self.showtext(f"please select a corret file img or viedo")
        else:
            self.showtext(f"set viedo {self.videofilepath}")

    # 获取文件路径
    def getfilepath(self):
        self.stop_videoorcamera()
        filepath = filedialog.askopenfilename(title="select your file")
        if filepath:
            return filepath

    # 切换检测状态
    def turn_detection(self):
        if self.detec:
            self.showtext("stop detection")
            self.detec = None
        else:
            self.showtext("start detection")
            self.detec = True

    # 在文本框中显示消息
    def showtext(self, text):
        self.text_frame.insert(tk.END, f"(yolo): ", "GUI")
        self.text_frame.tag_config("GUI", foreground='red')
        self.text_frame.insert(tk.END, f'{text}\n')
        self.text_frame.see(tk.END)
        return None

    # 打开摄像头
    def start_camera(self):
        if not self.iscamera:
            self.cap = cv2.VideoCapture(0)
            if self.cap:
                self.showtext("start camera")
                self.iscamera = True
                self.ispause = False
                self.update_frame()

    # 开始播放视频
    def start_video(self):
        if not self.playing:
            if self.videofilepath:
                self.cap = cv2.VideoCapture(self.videofilepath)
            elif self.imgpath:
                self.cap = cv2.VideoCapture(self.imgpath)
                self.detec = True
            if self.cap:
                self.showtext("start playing the video")
                self.playing = True
                self.ispause = False
                self.update_frame()

    # 暂停播放视频
    def pausevideo(self):
        if self.ispause:
            self.ispause = None
            self.update_frame()
        else:
            self.ispause = True

    # 停止视频或摄像头
    def stop_videoorcamera(self):
        if self.playing:
            self.showtext("stop playing the video")
            self.playing = False
            self.cap.release()
            self.label.config(image='')
        elif self.iscamera:
            self.showtext("stop camera")
            self.iscamera = False
            self.cap.release()
            self.label.config(image='')

    # 更新视频帧
    def update_frame(self):
        if self.playing or self.iscamera:
            ret, frame = self.cap.read()
            if ret and not self.ispause:
                # 调整帧的大小
                frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
                if self.detec:
                    frame, classes = detecion(frame, model=self.model, label=self.labels)
                    classes = list(set(classes))
                    self.showtext(f'objects:{" ".join(str(item) for item in classes)}')
                # 将帧从BGR转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.config(image=imgtk)
                self.label.after(33, self.update_frame)
            elif ret:
                if self.ispause:
                    self.showtext("viedo pauesd")

# 创建根窗口并启动应用程序
root = tk.Tk()
player = VideoPlayer(root)
root.mainloop()
