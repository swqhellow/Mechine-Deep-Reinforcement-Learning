import tkinter as tk
from tkinter import Label, Button, Frame
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from tkinter import filedialog
import re
import os
VIEDO_EXTENSIONS = ['mp4', 'avi', 'mov',
                    'mkv', 'flv', 'wmv', 'webm', 'mpeg', '3gp']
IMAGE_EXTENSIONS = ['jpg', 'png']
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
def loadcocolabel(label_path):
    label = []
    with open(label_path, 'r', encoding='utf-8')as file:
        for line in file:
            line = line.replace('"', '').replace(
                ',', '').replace('\n', '').split('.')
            line = re.sub(r'\(.*?\)', '', line[1]).strip()
            label.append(line)
    return label


def detecion(img, model, label):
    pre = model(img)
    boxes = pre[0].boxes
    classes = []
    for box, cls in zip(boxes.xyxy, boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[(cls%9)], 2)
        cv2.putText(img, f'{label[cls]}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, [255, 0, 0], 2)
        classes.append(label[cls])
    return img, classes


def get_defaultvideo():
    files = os.listdir()
    for file in files:
        ext = file.split('.', -1)[-1]
        if ext in VIEDO_EXTENSIONS:
            return file


class VideoPlayer:
    def __init__(self, root):
        self.model = YOLO(r'Learning\deep_learning\yoloGUI\yolov8n.pt')
        self.labels = loadcocolabel(
            r'Learning\deep_learning\yoloGUI\lable.txt')

        self.root = root
        self.root.title("YOLO GUI")
        self.root.geometry("900x600")

        # 创建上部分的视频显示框架
        self.video_frame = Frame(root)
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 创建Label用于显示视频帧
        self.label = Label(self.video_frame)
        self.label.pack()
        # 创建一个文本显示框
        self.text_frame = tk.Text(
            root, height=10, width=40, font=("Arial", 12))
        self.text_frame.pack(padx=10)
        # insert the initial words
        self.showtext("This is a detection GUI")

        # creat the buttom area in  the bottom
        self.controls_frame = Frame(root)
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.selectfile = Button(
            self.controls_frame, text="select viedo or img", command=self.get_viedopath)
        self.selectfile.pack(side=tk.LEFT, padx=10, pady=10)

        self.opencamera = Button(
            self.controls_frame, text='open camera', command=self.start_camera)
        self.opencamera.pack(side=tk.LEFT, padx=10, pady=10)

        # 添加“开始”和“结束”按钮
        self.start_button = Button(
            self.controls_frame, text="start", command=self.start_video)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.pausebutton = Button(
            self.controls_frame, text="pause", command=self.pausevideo)
        self.pausebutton.pack(side=tk.LEFT, padx=10, pady=10)

        self.stop_button = Button(
            self.controls_frame, text="end", command=self.stop_videoorcamera)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=10)
        # 添加检测按钮
        self.det_button = Button(
            self.controls_frame, text="detection", command=self.turn_detection)
        self.det_button.pack(side=tk.LEFT, padx=10, pady=10)

        # 初始化视频播放控制变量
        self.cap = None
        self.playing = False
        self.ispause = None
        self.detec = None
        self.videofilepath = get_defaultvideo()
        self.showtext(f'load default viedo:{self.videofilepath}')
        self.imgpath = None
        self.iscamera = None

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

    def getfilepath(self):
        self.stop_videoorcamera()
        filepath = filedialog.askopenfilename(title="select your file")
        if filepath:
            return filepath

    def turn_detection(self):
        if self.detec:
            self.showtext("stop detection")
            self.detec = None
        else:
            self.showtext("start detection")
            self.detec = True

    def showtext(self, text):
        self.text_frame.insert(
            tk.END, f"(yolo): ", "GUI")
        self.text_frame.tag_config("GUI", foreground='red')
        self.text_frame.insert(tk.END, f'{text}\n')
        self.text_frame.see(tk.END)
        return None

    def start_camera(self):
        if not self.iscamera:
            self.cap = cv2.VideoCapture(0)
            if self.cap:
                self.showtext("start camera")
                self.iscamera = True
                self.ispause = False
                self.update_frame()

    def start_video(self):
        if not self.playing:
            if self.videofilepath:
                self.cap = cv2.VideoCapture(self.videofilepath)  #
            elif self.imgpath:
                self.cap = cv2.VideoCapture(self.imgpath)  #
                self.detec = True
            if self.cap:
                self.showtext("start playing the video")
                self.playing = True
                self.ispause = False
                self.update_frame()

    def pausevideo(self):
        if self.ispause:
            self.ispause = None
            self.update_frame()
        else:
            self.ispause = True

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

    def update_frame(self):
        if self.playing or self.iscamera:
            ret, frame = self.cap.read()
            if ret and not self.ispause:
                frame = cv2.resize(
                    frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
                if self.detec:
                    frame, classes = detecion(
                        frame, model=self.model, label=self.labels)
                    classes = list(set(classes))
                    self.showtext(
                        f'objects:{" ".join(str(item) for item in classes)}')
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.config(image=imgtk)
                self.label.after(33, self.update_frame)
            elif ret:
                if self.ispause:
                    self.showtext("viedo pauesd")


root = tk.Tk()

player = VideoPlayer(root)

root.mainloop()
