import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

image = np.zeros((1000, 1000, 3), dtype=np.uint8)  # Creating a black image with OpenCV

# 读取image
def select_image():
    global image
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        # cv2.imshow("original_img", cv2.resize(image, (int(w * ratio), int(h * ratio))))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# 创建用于GUI展示的image并调整其大小以适应界面
def show_image(img):
    #global image
    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        h, w, _ = img.shape

    ratio = min(1000 / h, 1000 / w)
    cv2.imshow("img", cv2.resize(img, (int(w * ratio), int(h * ratio))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # image_show = cv2.resize(img, (int(w * ratio), int(h * ratio)))
    # label.config(image=image_show)
    # label.image = image_show

# 边缘检测
def detect_edges():
    global image
    global edge
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 100, 200)

def trigger():
    select_image()
    show_image(image)
    detect_edges()
    show_image(edge)


# 创建主窗口
root = tk.Tk()
root.title("Lego元件计数器")

# 创建标签以显示图片
label = tk.Label(root)
label.pack(padx=10, pady=10)

# 创建按钮以选择图片
select_button = tk.Button(root, text="请选择图片", command=trigger)
select_button.pack(padx=10, pady=5)

# 运行主循环
root.mainloop()
