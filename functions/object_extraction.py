import paddlers as pdrs
import cv2 as cv
import numpy as np
import time


def load_object_extraction(model_path):
    predictor = pdrs.deploy.Predictor(model_path)
    return predictor


def object_extraction(pic_path, predictor):
    im = cv.imread(pic_path)
    t1 = time.time()
    result = predictor.predict(im)
    t2 = time.time()
    period = t2 - t1

    result = result["label_map"]
    result = np.array(result, dtype=np.uint8)
    result = result * 255

    return result, period


# model = load_object_extraction('../weights/object_extraction')
# res, period = object_extraction(model)
# cv.imshow('test', res)
