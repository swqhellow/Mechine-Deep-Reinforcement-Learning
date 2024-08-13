import tkinter as tk
from tkinter import ttk

root = tk.Tk()

style = ttk.Style()
def hello():
    print('hellow')
# 设置全局按钮样式
style.configure('TButton', font=('Helvetica', 12), foreground='red', padding=10)

# 创建按钮并应用样式
button1 = ttk.Button(root, text='Button 1',command=hello)
button1.pack(pady=10)

# 创建一个自定义样式按钮
style.configure('My.TButton', font=('Helvetica', 14), foreground='blue', background='yellow')
button2 = ttk.Button(root, text='Button 2', style='My.TButton')
button2.pack(pady=10)

root.mainloop()
