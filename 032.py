import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像和模板
img = cv2.imread('bubbles.jpeg')
img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('template.png', 0)
result = img

#记录识别到的目标的数量
circle_num = 0

# 记录已检测区域的列表
detected_areas = []

#模板图片缩放尺度
scale_factors = range(6, 101, 1)

# 在不同尺度下进行模板匹配
for scale in scale_factors:
    scale = scale / 100

    scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
    res = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7

    locations = np.where(res >= threshold)
    for loc in zip(*locations[::-1]):
        w, h = scaled_template.shape[::-1]
        top_left = loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # 检查检测到的气泡区域是否与之前已检测的区域重叠，如果重叠则跳过此次检测
        overlapping = False
        for area in detected_areas:
            if (top_left[0] < area[2] and bottom_right[0] > area[0] and top_left[1] < area[3] and bottom_right[1] > area[1]):
                overlapping = True
                break
        #如果没有重叠，则把新区域加入到已检测区域的列表
        if not overlapping:
            detected_areas.append((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

            # 在原始图像上绘制矩形框显示匹配位置
            circle_num += 1
            cv2.rectangle(result, top_left, bottom_right, (0, 0, 250), 1)
            cv2.putText(result, '%.2f' % (scale), top_left, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                        color=(0, 0, 0), thickness=1)

# Matplotlib options
print(circle_num)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])

#Show the image
plt.show()
