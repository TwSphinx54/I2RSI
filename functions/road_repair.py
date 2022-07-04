import numpy as np
import cv2 as cv
from skimage import morphology

'''
1.图像切割
2.自适应阈值对道路进行粗略补全
  原理：假定区域内所有路网相连，判断标准：每个图形中只能有一个连通区域，即道路都是相连的
3.利用骨架提取进行修补区域定位
  原理：利用图像骨架定位出待修补区域的掩膜，再进行修补操作
'''


def roads_repair(image_path, score_path, coord):
    print('---------------- 开始补全 ----------------')
    image = cv.imread(image_path)
    score_map = np.load(score_path)
    image = image[coord[1]:coord[3], coord[0]:coord[2]]
    score_map = score_map[coord[1]:coord[3], coord[0]:coord[2]]

    num = 0.5
    ret = 10
    '''
    自适应阈值对道路进行粗略补全
    '''
    while ret > 2:
        label_map = (score_map > num).astype('uint8')
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[1] = [255, 255, 255]
        res = lut[label_map]
        res = morphology.remove_small_objects(res.astype(bool), 200, connectivity=2)  # 去除小区域
        res = morphology.remove_small_holes(res.astype(bool), 200, connectivity=2)  # 去除小洞
        res = res.astype('uint8') * 255
        res = cv.erode(res, np.ones((2, 2), np.uint8))  # 对图像做腐蚀
        ret, labels, stats, centroids = cv.connectedComponentsWithStats(res[:, :, 0], 4)  # 连通域检测
        num -= 0.01
        if ret == 2:  # 当直检测出两个连通域时，停止循环，再多一步提升补全效果
            num -= 0.03
            label_map = (score_map > num).astype('uint8')
            lut = np.zeros((256, 3), dtype=np.uint8)
            lut[1] = [255, 255, 255]
            res = lut[label_map]
            res = morphology.remove_small_objects(res.astype(bool), 500, connectivity=2)  # 去除小区域
            res = morphology.remove_small_holes(res.astype(bool), 200, connectivity=2)  # 去除小洞
            res = res.astype('uint8') * 255
            res = cv.erode(res, np.ones((2, 2), np.uint8))  # 对图像做腐蚀
            break
    '''
    利用骨架提取进行修补区域定位
    '''
    res_split = res[:, :, 1]
    res_split[res[:, :, 1] == 255] = 1
    image_split = image[:, :, 1]
    image_split[image[:, :, 1] == 255] = 1
    # 骨架提取
    skeleton0 = morphology.skeletonize(res_split)
    res_skeleton = skeleton0.astype(np.uint8) * 255
    skeleton0 = morphology.skeletonize(image_split)
    img_skeleton = skeleton0.astype(np.uint8) * 255
    # 对骨架进行膨胀操作
    res_dilate = cv.dilate(res_skeleton, np.ones((5, 5), np.uint8))
    img_dilate = cv.dilate(img_skeleton, np.ones((5, 5), np.uint8))
    # 进行图像相减操作，检测出现变化的区域
    res_sub = cv.subtract(res_dilate, img_dilate)
    res_sub = morphology.remove_small_objects(res_sub.astype(bool), 100, connectivity=2)  # 去除小区域
    res_sub = res_sub.astype('uint8') * 255
    res_sub = cv.dilate(res_sub, np.ones((3, 3), np.uint8))  # 图像膨胀，减少偏移带来的影响
    # 修补定位：利用相减的结果与粗修补结果相与得到修补的结果，再将修补的结果与原图相或进行修补区域定位
    res_sub = cv.bitwise_and(res_sub, res[:, :, 0])
    res_img = cv.bitwise_or(res_sub, image[:, :, 0])
    print('---------------- 补全完成 ----------------')
    return res_img


# road = roads_repair('../webpage/res/result.png', '../webpage/res/score.npy', [700, 700, 900, 900])
# cv.imwrite('road.png', road)
