import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

image = Image.new("RGB", (100, 100))

def select_image():
    global image
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)


def showing_image():
    global image
    image_show = image.copy()
    image_show.thumbnail((300, 300))  # 调整图像大小以适应界面
    photo = ImageTk.PhotoImage(image_show)
    label.config(image=photo)
    label.image = photo

def trigger():
    select_image()
    showing_image()
    
# 创建主窗口
root = tk.Tk()
root.title("Lego元件计数器")

# 创建标签以显示图片
label = tk.Label(root)
label.pack(padx=10, pady=10)

# 创建按钮以选择图片
select_button = tk.Button(root, text="请选择图片", command=trigger)
select_button.pack(padx=10, pady=5)

#cv2.imshow('Image', image)

# 运行主循环
root.mainloop()

