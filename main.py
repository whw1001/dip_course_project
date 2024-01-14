import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

image = np.zeros((1000, 1000, 3), dtype=np.uint8)  # Creating a black image with OpenCV

# 读取image
def select_image(path):
    global image
    #file_path = filedialog.askopenfilename()
    file_path = path
    #file_path = '30012.png'
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        # cv2.imshow("original_img", cv2.resize(image, (int(w * ratio), int(h * ratio))))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def denoise(img):
    #global denoised_image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.equalizeHist(img)
    denoised_image = cv2.GaussianBlur(img, (9, 9), 0)
    return denoised_image

# 创建用于GUI展示的image并调整其大小以适应界面
def show_image(img):

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
def detect_edges(img):
   # global edge
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(img, 15, 20)
    return edge

def segement(color, edge):
    rect = color.copy()
    # edges = cv2.Canny(image, 30, 100)

    # cv2.imwrite('edge.png', edges)

    # 寻找轮廓
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 过滤轮廓
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 75.0:  # 根据实际情况调整面积阈值
            filtered_contours.append(contour)

    # 切分图像
    segmented_images = []
    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        roi = color[y:y + h, x:x + w]
        segmented_images.append(roi)
        cv2.rectangle(rect, [x, y], [x + w, y + h], (0, 0, 250), 5)
        #cv2.imwrite(f"seg/segmented_image_{i}.png", roi)

    cv2.imwrite('rect.png', rect)
    return rect

#def trigger():
select_image('images/test_images/1.jpg')
#show_image(image)

denoised_image = denoise(image)
#show_image(denoised_image)

edge = detect_edges(denoised_image)
show_image(edge)
cv2.imwrite('canny6.png', edge)

rect = segement(image, edge)
show_image(rect)



# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# img_contrast = clahe.apply(img)

# # 创建主窗口
# root = tk.Tk()
# root.title("Lego元件计数器")
#
# # 创建标签以显示图片
# label = tk.Label(root)
# label.pack(padx=10, pady=10)
#
# # 创建按钮以选择图片
# select_button = tk.Button(root, text="请选择图片", command=trigger)
# select_button.pack(padx=10, pady=5)
#
# # 运行主循环
# root.mainloop()

#trigger()
