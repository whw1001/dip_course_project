import cv2
import numpy as np

image = np.zeros((1000, 1000, 3), dtype=np.uint8)

# 读取image
def select_image(path):
    global image
    file_path = path
    if file_path:
        image = cv2.imread(file_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

def show_image(img, name):

    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        h, w, _ = img.shape

    ratio = min(1000 / h, 1000 / w)
    cv2.imshow(f"{name}", cv2.resize(img, (int(w * ratio), int(h * ratio))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_color(image, color_name, color_dict):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound, upper_bound = color_dict[color_name]
    mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def find_contours(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 寻找轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 过滤轮廓
    filtered_contours = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 100:
            filtered_contours.append(contour)

    return filtered_contours

# 绘制轮廓
def draw_contours(img, contours):
    cns = img.copy()
    cv2.drawContours(cns, contours, -1, (0, 0, 0), 5)
    return cns


# 切分图像
def segement(original_image, mask, filtered_contours, current_color):
    #global segmented_images
    rect = original_image.copy()
    segmented_images = []
    j = 0
    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > 40000:
            roi = mask[y:y + h, x:x + w]
            segmented_images.append(roi)
            cv2.rectangle(rect, [x, y], [x + w, y + h], (0, 0, 250), 5)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 9)
            cv2.imwrite(f"segments/{current_color}_{j}.png", gray)
            j += 1

    cv2.imwrite('rect.png', rect)

    return rect, segmented_images

def sift(img, template):
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(template, None)

    # FLANN参数
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    # 创建FLANN匹配器
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用KNN匹配描述符
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
    if len(matches) == 0:
        return 0

    else :
    # 应用比例测试
        good_matches = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good_matches.append(m)

        match_rate = len(good_matches) / len(matches) * 100
        #print({color}, match_rate)
        return match_rate

# 定义颜色字典
colors = {
    'red': ([0, 40, 100], [6, 255, 255]),
    'yellow': ([19, 40, 140], [28, 255, 255]),
    'orange': ([10, 137, 100], [18, 255, 255]),
    'coffee': ([15, 40, 70], [21, 110, 200]),
    'blue': ([96, 40, 20], [106, 255, 255]),
    'dark blue': ([106, 40, 20], [114, 255, 255]),
    'light green': ([39, 20, 20], [44, 255, 255]),
    'green': ([71, 40, 60], [74, 255, 255]),
    'dark green': ([74, 25, 20], [82, 255, 120]),
    'pink': ([160, 40, 20], [167, 255, 255]),
    'black': ([0, 0, 20], [180, 40, 50]),
    'purple': ([145, 50, 50], [155, 255, 255]),
    'white': ([7, 8, 100], [17, 15, 255]),
    'gray': ([0, 0, 20], [160, 40, 128]),
}


select_image('test_images/10.jpg')

templates_paths = {
    #'1x4': 'templates/1x4.png',
    '2x4': 'templates/2x4.png',
    #'2x6': 'templates/2x6.png',
    #'wheel': 'templates/wheel.png'
}

templates = {}

#读取模板图像
for templates_path in templates_paths.values():
    template = cv2.imread(templates_path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    templates[templates_path] = template

templates_length = len(templates)
counts = np.zeros(templates_length)

for color in colors:

    mask = extract_color(image, color, colors)
    #show_image(mask, color)

    contours = find_contours(mask)

    cns = draw_contours(image, contours)
    #show_image(cns, '{color}')

    rect, segmented_images = segement(image, mask, contours, color)
    #show_image(rect, 'rect')

    for segmented_image in segmented_images:
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        #_, thresh1 = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #show_image(thresh1, color)

        K = 0
        for template in templates.values():
            #_, thresh2 = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #match_rate = sift(thresh1, thresh2)
            match_rate = sift(segmented_image, template)
            if match_rate >= 27:
                counts[K] += 1
            K += 1

print(counts)

