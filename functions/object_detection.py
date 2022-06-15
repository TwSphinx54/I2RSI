import paddlers as pdrs
from paddlers import transforms as T
import cv2 as cv
import time


def load_object_detection(model_path):
    predictor = pdrs.deploy.Predictor(model_path)
    return predictor


def cal_coord(rate, bbox):
    r_h, r_w = rate
    i_bbox = []
    r_bbox = []
    for k in bbox:
        x_min, y_min, w, h = k
        x_max = int((x_min + w) / r_w)
        y_max = int((y_min + h) / r_h)
        x_min = int(x_min / r_w)
        y_min = int(y_min / r_h)
        i_bbox.append([x_min, y_min, x_max, y_max])
    return i_bbox


def draw_box_corner(draw_img, bbox, length, corner_color):
    # Top Left
    cv.line(draw_img, (bbox[0], bbox[1]), (bbox[0] + length, bbox[1]), corner_color, thickness=3)
    cv.line(draw_img, (bbox[0], bbox[1]), (bbox[0], bbox[1] + length), corner_color, thickness=3)
    # Top Right
    cv.line(draw_img, (bbox[2], bbox[1]), (bbox[2] - length, bbox[1]), corner_color, thickness=3)
    cv.line(draw_img, (bbox[2], bbox[1]), (bbox[2], bbox[1] + length), corner_color, thickness=3)
    # Bottom Left
    cv.line(draw_img, (bbox[0], bbox[3]), (bbox[0] + length, bbox[3]), corner_color, thickness=3)
    cv.line(draw_img, (bbox[0], bbox[3]), (bbox[0], bbox[3] - length), corner_color, thickness=3)
    # Bottom Right
    cv.line(draw_img, (bbox[2], bbox[3]), (bbox[2] - length, bbox[3]), corner_color, thickness=3)
    cv.line(draw_img, (bbox[2], bbox[3]), (bbox[2], bbox[3] - length), corner_color, thickness=3)


def object_detection(pic_path, predictor):
    INPUT_SIZE = 608
    threshold = 0.4
    alpha = 0.8
    gamma = 1
    line_color = (255, 0, 255)
    corner_color = (0, 255, 0)

    eval_transforms = T.Compose([
        T.Resize(target_size=INPUT_SIZE, interp='CUBIC'),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    im = cv.imread(pic_path)
    im_r = cv.resize(im[..., ::-1], (INPUT_SIZE, INPUT_SIZE), interpolation=cv.INTER_CUBIC)
    t1 = time.time()
    pred = predictor.predict(im_r, eval_transforms)
    t2 = time.time()

    r_h = INPUT_SIZE / im.shape[0]
    r_w = INPUT_SIZE / im.shape[1]

    bbox = [k['bbox'] for k in pred]
    score = [k['score'] for k in pred]

    a_bbox = []
    a_score = []
    for k in range(len(bbox)):
        if score[k] > threshold:
            a_score.append(score[k])
            a_bbox.append(bbox[k])

    coords = cal_coord([r_h, r_w], a_bbox)
    im_fill = im.copy()
    for coord in coords:
        im = cv.rectangle(im, (coord[0], coord[1]), (coord[2], coord[3]), color=line_color, thickness=2)
        im_fill = cv.rectangle(im_fill, (coord[0], coord[1]), (coord[2], coord[3]), color=line_color, thickness=-1)
        corner_length = int(min(coord[3] - coord[1], coord[2] - coord[0]) / 5)
        draw_box_corner(im, coord, corner_length, corner_color)
    im_out = cv.addWeighted(im, alpha, im_fill, 1 - alpha, gamma)

    return im_out, coords, a_score, t2 - t1
