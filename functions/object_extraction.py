import paddlers as pdrs
import numpy as np
import time
import cv2 as cv
from functions.object_classification import add_alpha


def load_object_extraction(model_path):
    predictor = pdrs.deploy.Predictor(model_path)
    return predictor


def oe_mix(img, road):
    mixed = cv.addWeighted(img, 0.3, road, 0.7, 1)
    mixed = cv.addWeighted(img, 0.3, mixed, 0.7, 1)
    return mixed


def object_extraction(img_path, predictor):
    img = cv.imread(img_path)
    t1 = time.time()
    result = predictor.predict(img)
    t2 = time.time()
    period = t2 - t1

    score_map = result['score_map']
    label_map = result['label_map']

    res = np.array(label_map, dtype=np.uint8)
    res = res * 255

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[1] = [1, 0, 255]
    res_img = lut[label_map]
    roads = add_alpha(res_img)
    scores = sum(map(sum, score_map[label_map == 1])) / (len(score_map[label_map == 1]) + 1)

    img_o = img.copy()
    img[label_map == 1] = [1, 0, 255]
    mixed = oe_mix(img_o, img)

    '''
    IoU为0.59，Acc为0.78，Kappa系数为0.72, F1为0.74
    '''
    return res, roads, scores, period, mixed, score_map
