import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image.thumbnail((300, 300))  # 调整图像大小以适应界面
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

# 创建主窗口
root = tk.Tk()
root.title("图片选择器")

# 创建标签以显示图片
label = tk.Label(root)
label.pack(padx=10, pady=10)

# 创建按钮以选择图片
select_button = tk.Button(root, text="选择图片", command=select_image)
select_button.pack(padx=10, pady=5)

# 运行主循环
root.mainloop()
