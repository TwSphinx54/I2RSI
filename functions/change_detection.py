import cv2 as cv
import numpy as np
import time
from skimage import morphology
from functions.object_classification import add_alpha


def repair(img, o_threshold, h_threshold):
    img_t = morphology.remove_small_objects(img.astype(bool), o_threshold, connectivity=2)
    img_t = morphology.remove_small_holes(img_t.astype(bool), h_threshold, connectivity=2)
    return img_t.astype(int) * 255


def change_detection(A, B, predictor, o_thres, h_thres):
    t1 = time.time()
    result = predictor.predict(img_file=(A, B))[0]
    t2 = time.time()
    period = t2 - t1

    result_map = result["label_map"]
    score_map = result['score_map']

    result = np.array(result_map, dtype=np.uint8)
    result = result * 255
    result = repair(result, o_thres, h_thres)
    # result = repair(result, 200, 1000)
    label = (result / 255).astype(int)
    score = sum(map(sum, score_map[label == 1])) / (len(score_map[label == 1]) + 1)

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[1] = [1, 0, 255]
    bgr_res = lut[label]
    alpha = add_alpha(bgr_res)

    img_a = cv.imread(A)
    img_b = cv.imread(B)
    img_bo = img_b.copy()
    img_b[label == 1] = [1, 0, 255]
    mixed = cv.addWeighted(img_bo, 0.5, img_b, 0.5, 1)
    mixed = cv.addWeighted(img_a, 0.5, mixed, 0.5, 1)

    return result, alpha, score, period, mixed


def change_detection_en(A, B, predictor, predictor_en, o_thres, h_thres):
    t1 = time.time()
    result = predictor.predict(img_file=(A, B))[0]
    result_en = predictor_en.predict(img_file=(A, B))[0]
    t2 = time.time()
    period = t2 - t1

    # result_map = result['label_map']
    score_map = (result['score_map'] + result_en['score_map']) / 2
    result_map = score_map > 255

    result = np.array(result_map, dtype=np.uint8)
    result = result * 255
    result = repair(result, o_thres, h_thres)
    # result = repair(result, 200, 1000)
    label = (result / 255).astype(int)
    score = sum(map(sum, score_map[label == 1])) / (len(score_map[label == 1]) + 1)

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[1] = [1, 0, 255]
    bgr_res = lut[label]
    alpha = add_alpha(bgr_res)

    img_a = cv.imread(A)
    img_b = cv.imread(B)
    img_bo = img_b.copy()
    img_b[label == 1] = [1, 0, 255]
    mixed = cv.addWeighted(img_bo, 0.5, img_b, 0.5, 1)
    mixed = cv.addWeighted(img_a, 0.5, mixed, 0.5, 1)

    return result, alpha, score, period, mixed
