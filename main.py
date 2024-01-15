import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

image = np.zeros((1000, 1000, 3), dtype=np.uint8)  # Creating a black image with OpenCV

# 读取image
def select_image(path):
    global image
    file_path = path
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape


def denoise(img):
    #global denoised_image
    # vimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.equalizeHist(img)
    denoised_image = cv2.GaussianBlur(img, (3, 3), 0)
    return denoised_image

def show_image(img, name):

    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        h, w, _ = img.shape

    ratio = min(1000 / h, 1000 / w)
    cv2.imshow(f"{name}", cv2.resize(img, (int(w * ratio), int(h * ratio))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_shadow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 定义阴影的HSV范围
    lower_shadow = np.array([0, 0, 115], dtype=np.uint8)
    upper_shadow = np.array([255, 33, 207], dtype=np.uint8)
    # 创建阴影掩码
    shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)
    # 去除阴影
    img_no_shadow = cv2.bitwise_and(img, img, mask=~shadow_mask)
    # 将阴影区域改为白色
    img_no_shadow[~shadow_mask == 0] = [255, 255, 255]

    return img_no_shadow

def extract_color(image, color_name, color_dict):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound, upper_bound = color_dict[color_name]
    mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def detect_edges(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(img, 'nnn ')
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edge = cv2.Canny(img, 100, 200)
   # # 霍夫变换
   #  lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=25)
   #
   #  # 绘制连接的线条到空白图像
   #  connected_edge = np.zeros_like(img)
   #  for line in lines:
   #      x1, y1, x2, y2 = line[0]
   #      cv2.line(connected_edge, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return edge

def find_contours(edge):
    # 寻找轮廓
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤轮廓
    filtered_contours = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 10000:
            filtered_contours.append(contour)

    return contours

def draw_contours(img, contours):
    cv2.drawContours(img, contours, -1, (0, 0, 0), 5)

# 切分图像
def segement(original_image, filtered_contours):
    rect = original_image.copy()
    segmented_images = []
    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > 50000:
            roi = original_image[y:y + h, x:x + w]
            segmented_images.append(roi)
            cv2.rectangle(rect, [x, y], [x + w, y + h], (0, 0, 250), 5)
        #cv2.imwrite(f"seg/segmented_image_{i}.png", roi)

    cv2.imwrite('rect.png', rect)

    return rect


# 定义颜色字典
color_thresholds = {
    'red': ([0, 100, 100], [10, 255, 255]),
    'yellow': ([20, 100, 100], [30, 255, 255]),
    'blue': ([100, 100, 100], [120, 255, 255]),
    'green': ([70, 40, 40], [80, 255, 255]),
    'purple': ([130, 50, 50], [160, 255, 255])
}


select_image('images/test_images/3.jpg')

green_result = extract_color(image, 'green', color_thresholds)
show_image(green_result, 'green')

#green_result = cv2.cvtColor(green_result, cv2.COLOR_HSV2GRAY)
edge = detect_edges(green_result)
show_image(edge, 'edge')

green_result = cv2.cvtColor(green_result, cv2.COLOR_BGR2GRAY)
contours = find_contours(green_result)
rect = segement(image, contours)
show_image(rect, 'rect')

