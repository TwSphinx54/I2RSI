import cv2 as cv
import numpy as np


def roads_repair(image_path):
    image = cv.imread(image_path)
    repair_points1 = [  # 可以按照此格式添加任意修补区域
        [[53, 256], [62, 246], [109, 311], [117, 303]],
        [[279, 400], [292, 389], [286, 415], [304, 397]],
        [[1, 922], [3, 945], [114, 1043], [69, 987]],
        [[629, 1440], [568, 1467], [548, 1498], [611, 1429]],
        [[545, 1050], [551, 1054], [546, 1058], [540, 1054]],
        [[1074, 1186], [1084, 1196], [976, 1314], [971, 1309]]
    ]
    # 填充闭合区域
    for point in repair_points1:
        pts = np.array(point)
        cv.fillPoly(image, [pts], color=(255, 255, 255))  # 此处可以修改颜色，白色[255,255,255]、红色[255，0，0]
    # # 按照这个思路还可以添加擦除功能，有部分铁路被误认为是公路
    # repair_points2 = [  # 可以按照此格式添加任意修补区域
    #     [[660, 736], [691, 751], [551, 896], [527, 870]]
    # ]
    # # 填充闭合区域
    # for point in repair_points2:
    #     pts = np.array(point)
    #     cv.fillPoly(image, [pts], color=(0, 0, 0, 0))  # 此处可以修改颜色，白色[255,255,255]、红色[255，0，0]
    # cv.imwrite("2.png", image)
    return image
