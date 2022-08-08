import numpy as np
import cv2 as cv
import time


def split_type(label_map, num):  # 分类别
    lut = np.zeros((256, 3), dtype=np.uint8)
    if num == 0:
        lut[num] = [1, 0, 255]
    elif num == 1:
        lut[num] = [142, 255, 30]
    elif num == 2:
        lut[num] = [255, 0, 60]
    elif num == 3:
        lut[num] = [1, 222, 255]
    # lut[num] = [255, 222, 255]
    types = lut[label_map]
    return types


def morphology_open(img):  # 形态学开运算
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # OpenCV定义的结构矩形元素
    res_img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)  # 形态学开运算
    return res_img


def add_alpha(img):
    b_channel, g_channel, r_channel = cv.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # alpha通道每个像素点区间为[0,255], 0为完全透明，255是完全不透明
    alpha_channel[b_channel == 0] = 0
    img_BGRA = cv.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA


def split_img(label_map):  # 把label_map 按照类别拆分成4张二值图像并做开运算
    type0 = split_type(label_map, 0)
    type1 = split_type(label_map, 1)
    type2 = split_type(label_map, 2)
    type3 = split_type(label_map, 3)
    type0 = morphology_open(type0)
    type1 = morphology_open(type1)
    type2 = morphology_open(type2)
    type3 = morphology_open(type3)
    return type0, type1, type2, type3


def object_classification(img_path, predictor):
    # 类别预测
    img = cv.imread(img_path)
    t1 = time.time()
    result = predictor.predict(img)
    t2 = time.time()
    score_map = result['score_map']  # 得分结果
    label_map = result['label_map']  # 类别结果

    # 图像染色
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[0] = [0, 0, 255]
    lut[1] = [142, 255, 30]
    lut[2] = [255, 0, 60]
    lut[3] = [0, 222, 255]
    lut[4] = [0, 0, 0]
    res_img = lut[label_map]

    # 拆分图像
    # type0道路, type1树木, type2人造用地, type3空地
    type0, type1, type2, type3 = split_img(label_map)  # 按照类别拆分为4张二值图像
    types = [type0, type1, type2, type3]
    # 输出参数
    period = t2 - t1
    '''
    mIou = 0.36
    Iou = [0.11, 0.54, 0.18, 0.11, 0.85]
    OAcc = 0.85
    Acc = [0.49, 0.78, 0.60, 0.36, 0.87]
    F1 = [0.20, 0.70, 0.31, 0.20, 0.92]
    Kappa = 0.41
    '''
    scores = [0, 0, 0, 0]
    areas = [0, 0, 0, 0]
    for num in range(4):
        types[num] = add_alpha(types[num])
        scores[num] = sum(map(sum, score_map[types[num][:, :, 3] == 255])) / (
                len(score_map[types[num][:, :, 3] == 255]) + 1)
        areas[num] = len(types[num][types[num][:, :, 3] == 255])

    img_a = img.copy()
    img_a[label_map == 0] = [0, 0, 255]
    img_a[label_map == 1] = [142, 255, 30]
    img_a[label_map == 2] = [255, 0, 60]
    img_a[label_map == 3] = [0, 222, 255]
    mixed = cv.addWeighted(img, 0.3, img_a, 0.7, 1)
    mixed = cv.addWeighted(img, 0.3, mixed, 0.7, 1)

    return res_img, types, scores, period, areas, mixed
