import paddlers as pdrs
import cv2 as cv
import numpy as np
import time
from functions.object_classification import add_alpha


def load_change_detection(model_path):
    predictor = pdrs.deploy.Predictor(model_path)
    return predictor


def change_detection(A, B, predictor):
    t1 = time.time()
    result = predictor.predict(img_file=(A, B))[0]
    t2 = time.time()
    period = t2 - t1

    result_map = result["label_map"]
    score_map = result['score_map']
    score = sum(map(sum, score_map[result_map == 1])) / (len(score_map[result_map == 1]) + 1)

    result = np.array(result_map, dtype=np.uint8)
    result = result * 255

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[1] = [1, 0, 255]
    bgr_res = lut[result_map]
    alpha = add_alpha(bgr_res)

    img_a = cv.imread(A)
    img_b = cv.imread(B)
    alpha_channel = np.ones(img_a.shape[:2], dtype=img_a.dtype) * 125
    img_a_al = cv.merge((img_a, alpha_channel))
    img_b_al = cv.merge((img_b, alpha_channel))
    mixed = cv.add(img_a_al, img_b_al)
    mixed = cv.add(mixed, alpha)

    return result, alpha, score, period, mixed
