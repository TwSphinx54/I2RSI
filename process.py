import math
import os
import json
from flask import Flask, flash, request, redirect, url_for, render_template
from urllib.request import urlretrieve
from functions.object_detection import object_detection
from functions.object_classification import object_classification, add_alpha
from functions.object_extraction import load_model, object_extraction, oe_mix
from functions.change_detection import change_detection
from functions.road_repair import roads_repair
import cv2 as cv
import numpy as np
import shutil
import sqlite3
import random

UPLOAD_FOLDER = './webpage/res'
WEIGHT_FOLDER = './weights'
WEIGHT_DB = './weights/weights.db'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# MODELS = ['playground', 'aircraft', 'oiltank', 'overpass']

app = Flask(__name__, template_folder="./webpage", static_folder='./webpage', static_url_path="")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
pro, model, loc, score, period, shape, areas = [0] * 7
o_threshold, h_threshold = [200, 200]


def write_report(file_path, method, paras):
    with open(file_path, 'w') as f:
        if method == '0':
            score_p, period_p, shape_p, fn = paras
            f.write('<center>I2RSI遥感图像智能解译系统</center>\n\n')
            f.write('# 分析报告\n\n')
            f.write('## 基本信息\n\n')
            f.write('**解译方式**：目标提取\n\n')
            f.write('**文件名**：' + fn + '\n\n')
            f.write('**图像解译时间**：' + str(period_p) + 'ms\n\n')
            f.write('**图像分辨率**：' + str(shape_p[0]) + '×' + str(shape_p[1]) + '\n\n')
            f.write('## 检测结果\n\n')
            f.write('**提取道路平均得分**：' + str(score_p) + '\n')
        elif method == '1':
            score_p, period_p, shape_p, fn1, fn2 = paras
            f.write('<center>I2RSI遥感图像智能解译系统</center>\n\n')
            f.write('# 分析报告\n\n')
            f.write('## 基本信息\n\n')
            f.write('**解译方式**：变化检测\n\n')
            f.write('**文件名**：\n\n* ' + fn1 + '\n* ' + fn2 + '\n\n')
            f.write('**图像解译时间**：' + str(period_p) + 'ms\n\n')
            f.write('**图像分辨率**：' + str(shape_p[0]) + '×' + str(shape_p[1]) + '\n\n')
            f.write('## 检测结果\n\n')
            f.write('**变化部分平均得分**：' + str(score_p) + '\n')
        elif method == '2':
            loc_p, score_p, period_p, shape_p, fn = paras
            f.write('<center>I2RSI遥感图像智能解译系统</center>\n\n')
            f.write('# 分析报告\n\n')
            f.write('## 基本信息\n\n')
            f.write('**解译方式**：目标检测\n\n')
            f.write('**文件名**：' + fn + '\n\n')
            f.write('**图像解译时间**：' + str(period_p) + 'ms\n\n')
            f.write('**图像分辨率**：' + str(shape_p[0]) + '×' + str(shape_p[1]) + '\n\n')
            f.write('## 检测结果\n\n')
            for i in range(len(score_p)):
                f.write('---\n\n')
                f.write('### 目标' + str(i + 1) + '\n\n')
                f.write('**目标位于图像上的坐标**：' + str(loc[i]) + '\n\n')
                f.write('**目标得分**：' + str(score_p[i]) + '\n\n')
        elif method == '3':
            score_p, period_p, shape_p, areas_p, fn = paras
            f.write('<center>I2RSI遥感图像智能解译系统</center>\n\n')
            f.write('# 分析报告\n\n')
            f.write('## 基本信息\n\n')
            f.write('**解译方式**：变化检测\n\n')
            f.write('**文件名**：' + fn + '\n\n')
            f.write('**图像解译时间**：' + str(period_p) + 'ms\n\n')
            f.write('**图像分辨率**：' + str(shape_p[0]) + '×' + str(shape_p[1]) + '\n\n')
            f.write('## 检测结果\n\n')
            s = shape_p[0] * shape[1]
            for i in range(4):
                f.write('---\n')
                f.write('## 类别' + str(i + 1) + '\n\n')
                f.write('**类别得分**：' + str(score_p[i]) + '\n\n')
                f.write('**类别面积**：' + str(areas_p[i]) + '\n\n')
                f.write('**类别占比**：' + str(areas_p[i] / s * 100) + '%\n\n')
    return True


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clip_img(img, coord):
    return img[coord[1]:coord[3], coord[0]:coord[2]]


@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    if request.method == 'GET':
        animation = request.args.get('animation')
        return render_template('welcome.html', animation=animation)
    elif request.method == 'POST':
        global pro, model, o_threshold, h_threshold
        status = request.values['status']
        if status == 'model':
            con = sqlite3.connect(WEIGHT_DB)
            cur = con.cursor()
            f_type = request.values['f_type']
            od_type = request.values['od_type']
            q_res = cur.execute("SELECT id,name,desc FROM weights WHERE func=(?) AND od_class=(?)",
                                (f_type, od_type))
            res = {}
            for row in q_res:
                res[row[0]] = [row[1], row[2]]
            con.close()
            return json.dumps(res)
        elif status == 'load':
            con = sqlite3.connect(WEIGHT_DB)
            cur = con.cursor()
            pro = request.values['pro']
            model_id = int(request.values['model_id'])

            q_res = cur.execute("SELECT path FROM weights WHERE id=(?)", (model_id,))
            model_path = q_res.fetchone()[0]
            if pro == '1':
                o_threshold = int(request.values['o_thres'])
                h_threshold = int(request.values['h_thres'])

            model = load_model(model_path)
            con.close()
            return redirect(url_for('upload_file'))
        elif status == 'upload':
            pro = request.values['pro']
            pro_type = request.values['type']
            model_zip = request.files['model']
            model_name = request.values['name']
            model_desc = request.values['desc']

            save_path = WEIGHT_FOLDER + '/' + model_name.replace(' ', '_') + '/'
            zip_path = os.path.join(WEIGHT_FOLDER, model_zip.filename)
            model_zip.save(zip_path)
            shutil.unpack_archive(zip_path, save_path)
            os.remove(zip_path)
            if set(os.listdir(save_path)) == {'model.pdiparams', 'model.pdiparams.info', 'model.pdmodel', 'model.yml',
                                              'pipeline.yml'}:
                con = sqlite3.connect(WEIGHT_DB)
                cur = con.cursor()
                last_id = cur.execute("SELECT id FROM weights ORDER BY id desc limit 0,1").fetchone()[0]
                cur.execute(
                    "INSERT INTO weights VALUES ((?), (?), (?), (?), (?), (?))",
                    (last_id + 1, model_name, save_path, int(pro), model_desc, pro_type))
                con.commit()
                con.close()
                return 'DONE'
            else:
                shutil.rmtree(save_path)
                return 'WRONG'
        elif status == 'del':
            model_id = int(request.values['model_id'])
            con = sqlite3.connect(WEIGHT_DB)
            cur = con.cursor()
            rm_path = cur.execute("SELECT path FROM weights WHERE id=(?)", (model_id,)).fetchone()[0]
            shutil.rmtree(rm_path)
            cur.execute("DELETE FROM weights WHERE id=(?)", (model_id,))
            con.commit()
            con.close()
            return 'DONE'


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global loc, score, period, shape, areas
    if request.method == 'GET':
        return render_template('index.html', pro=pro)
    elif request.method == 'POST':
        status = request.values['status']
        if status == 'change':
            return redirect(url_for('welcome', animation=False))
        elif (status == 'upload') | (status == 'select'):
            if pro == '1':
                file1 = request.files['image1']
                file2 = request.files['image2']
                # If the user does not select a file, the browser submits an empty file without a filename.
                if (file1.filename == '') | (file2.filename == ''):
                    flash('No selected file')
                    return redirect(request.url)
                if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
                    filename1 = 'A.png'
                    filename2 = 'B.png'
                    A = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
                    B = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
                    file1.save(A)
                    file2.save(B)
                    res, alpha, score, period, mixed = change_detection(A, B, model, o_threshold,
                                                                        h_threshold)
                    shape = list(res.shape)
                    cv.imwrite(UPLOAD_FOLDER + '/result.png', res)
                    cv.imwrite(UPLOAD_FOLDER + '/change.png', alpha)
                    cv.imwrite(UPLOAD_FOLDER + '/mixed.png', mixed)
                    write_report(UPLOAD_FOLDER + '/report.md', pro,
                                 [score, period, shape, file1.filename, file2.filename])
                    return redirect(url_for('main_process'))
            else:
                filename = 'origin.png'
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if status == 'select':
                    urlretrieve(request.values['img_url'], save_path)
                    fn = request.values['img_url']
                else:
                    file = request.files['image']
                    file.save(save_path)
                    fn = file.filename
                if pro == '0':
                    res, roads, score, period, mixed, score_map = object_extraction(save_path, model)
                    shape = list(res.shape)
                    cv.imwrite(UPLOAD_FOLDER + '/result.png', res)
                    cv.imwrite(UPLOAD_FOLDER + '/roads.png', roads)
                    cv.imwrite(UPLOAD_FOLDER + '/mixed.png', mixed)
                    np.save(UPLOAD_FOLDER + '/score.npy', score_map[:, :, 1])
                    write_report(UPLOAD_FOLDER + '/report.md', pro, [score, period, shape, fn])
                elif pro == '2':
                    res, loc, score, period = object_detection(save_path, model)
                    shape = list(res.shape)
                    cv.imwrite(UPLOAD_FOLDER + '/result.png', res)
                    write_report(UPLOAD_FOLDER + '/report.md', pro, [loc, score, period, shape, fn])
                elif pro == '3':
                    res, types, score, period, areas, mixed = object_classification(save_path, model)
                    cv.imwrite(UPLOAD_FOLDER + '/result.png', res)
                    cv.imwrite(UPLOAD_FOLDER + '/mixed.png', mixed)
                    shape = list(res.shape)
                    for k in range(4):
                        cv.imwrite(UPLOAD_FOLDER + '/class' + str(k) + '.png', types[k])
                    write_report(UPLOAD_FOLDER + '/report.md', pro, [score, period, shape, areas, fn])
                return redirect(url_for('main_process'))
        elif status == 'multi':
            no = request.values['no']
            res_path = UPLOAD_FOLDER + '/result'
            if not os.path.exists(res_path):
                os.mkdir(res_path)
            if not os.path.exists(UPLOAD_FOLDER + '/multi'):
                os.mkdir(UPLOAD_FOLDER + '/multi')
            multi_path = UPLOAD_FOLDER + '/multi/' + no
            if not os.path.exists(multi_path):
                os.mkdir(multi_path)
            filename = 'origin.png'
            save_path = os.path.join(multi_path, filename)
            if status == 'select':
                urlretrieve(request.values['img_url'], save_path)
            else:
                file = request.files['image']
                file.save(save_path)
            if pro == '0':
                res, roads, score, period, mixed, score_map = object_extraction(save_path, model)
                shape = list(res.shape)
                cv.imwrite(multi_path + '/result.png', res)
                cv.imwrite(res_path + '/result' + no + '_' + str(random.random()) + '.png', res)
                cv.imwrite(multi_path + '/roads.png', roads)
                cv.imwrite(multi_path + '/mixed.png', mixed)
                np.save(UPLOAD_FOLDER + '/score.npy', score_map[:, :, 1])
                paras = {'score': score, 'period': period, 'shape': shape}
                paras_json = json.dumps(paras)
                with open(multi_path + '/paras.json', 'w') as file:
                    file.write(paras_json)
            elif pro == '2':
                res, loc, score, period = object_detection(save_path, model)
                shape = list(res.shape)
                cv.imwrite(multi_path + '/result.png', res)
                cv.imwrite(res_path + '/result' + no + '_' + str(random.random()) + '.png', res)
                paras = {'loc': loc, 'score': score, 'period': period, 'shape': shape}
                paras_json = json.dumps(paras)
                with open(multi_path + '/paras.json', 'w') as file:
                    file.write(paras_json)
            elif pro == '3':
                res, types, score, period, areas, mixed = object_classification(save_path, model)
                cv.imwrite(multi_path + '/result.png', res)
                cv.imwrite(res_path + '/result' + no + '_' + str(random.random()) + '.png', res)
                cv.imwrite(multi_path + '/mixed.png', mixed)
                shape = list(res.shape)
                for k in range(4):
                    cv.imwrite(multi_path + '/class' + str(k) + '.png', types[k])
                paras = {'score': score, 'period': period, 'shape': shape, 'areas': areas}
                paras_json = json.dumps(paras)
                with open(multi_path + '/paras.json', 'w') as file:
                    file.write(paras_json)
            is_end = request.values['is_end']
            if is_end == 'yes':
                shutil.make_archive(UPLOAD_FOLDER + '/result', 'zip', res_path)
                shutil.rmtree(res_path)
                return redirect(url_for('main_process', multi=True, no=no))
            else:
                return 'DONE'


@app.route('/result', methods=['GET', 'POST'])
def main_process():
    if request.method == 'GET':
        multi = request.args.get('multi')
        no = request.args.get('no')
        if pro == '1':
            return render_template('main.html', pro=pro, ele=[score, period, shape], multiF=multi, no=no)
        elif pro == '2':
            return render_template('main.html', pro=pro, ele=[loc, score, period, shape], multiF=multi, no=no)
        elif pro == '3':
            return render_template('main.html', pro=pro, ele=[score, period, shape, areas], multiF=multi, no=no)
        else:
            return render_template('main.html', pro=pro, ele=[score, period, shape], multiF=multi, no=no)
    elif request.method == 'POST':
        status = request.values['status']
        if status == 'change':
            return redirect(url_for('welcome', animation=False))
        elif status == 'back':
            return redirect(url_for('upload_file'))
        elif status == 'clip':
            coord = [request.values['x0'], request.values['y0'], request.values['x1'], request.values['y1']]
            coord = [math.floor(float(k)) for k in coord]
            img = cv.imread(UPLOAD_FOLDER + '/origin.png')
            clipped = clip_img(img, coord)
            cv.imwrite(UPLOAD_FOLDER + '/clip.png', clipped)
            return {'h': clipped.shape[0], 'w': clipped.shape[1]}
        elif status == 'select':
            coord = [request.values['x0'], request.values['y0'], request.values['x1'], request.values['y1']]
            coord = [math.floor(float(k)) for k in coord]
            if coord[2] < coord[0]:
                coord[0], coord[2] = coord[2], coord[0]
            if coord[3] < coord[1]:
                coord[1], coord[3] = coord[3], coord[1]
            img_a = cv.imread(UPLOAD_FOLDER + '/A.png')
            img_b = cv.imread(UPLOAD_FOLDER + '/B.png')
            img_a = clip_img(img_a, coord)
            img_b = clip_img(img_b, coord)
            cv.imwrite(UPLOAD_FOLDER + '/clipA.png', img_a)
            cv.imwrite(UPLOAD_FOLDER + '/clipB.png', img_b)
            return {'h': img_a.shape[0], 'w': img_a.shape[1]}
        elif status == 'preview':
            coord = [request.values['x0'], request.values['y0'], request.values['x1'], request.values['y1']]
            coord = [math.floor(float(k)) for k in coord]
            if coord[2] < coord[0]:
                coord[0], coord[2] = coord[2], coord[0]
            if coord[3] < coord[1]:
                coord[1], coord[3] = coord[3], coord[1]
            img = cv.imread(UPLOAD_FOLDER + '/mixed.png')
            img = clip_img(img, coord)
            cv.imwrite(UPLOAD_FOLDER + '/clipP.png', img)
            return {'h': img.shape[0], 'w': img.shape[1]}
        elif status == 'complete':
            road_o = cv.imread(UPLOAD_FOLDER + '/result.png')
            coord = [request.values['x0'], request.values['y0'], request.values['x1'], request.values['y1']]
            coord = [math.floor(float(k)) for k in coord]
            if coord[2] < coord[0]:
                coord[0], coord[2] = coord[2], coord[0]
            if coord[3] < coord[1]:
                coord[1], coord[3] = coord[3], coord[1]
            road = roads_repair(UPLOAD_FOLDER + '/result.png', UPLOAD_FOLDER + '/score.npy', coord)
            road_oc = clip_img(road_o, coord)
            ori = cv.imread(UPLOAD_FOLDER + '/origin.png')
            ori = clip_img(ori, coord)
            ori_o = ori.copy()
            ori[road == 255] = [142, 255, 30]
            ori[road_oc[:, :, 0] == 255] = [1, 0, 255]
            comp_prev = oe_mix(ori_o, ori)
            cv.imwrite(UPLOAD_FOLDER + '/clipP.png', comp_prev)
            return {'h': comp_prev.shape[0], 'w': comp_prev.shape[1]}
        elif status == 'confirm':
            coord = [request.values['x0'], request.values['y0'], request.values['x1'], request.values['y1']]
            coord = [math.floor(float(k)) for k in coord]
            if coord[2] < coord[0]:
                coord[0], coord[2] = coord[2], coord[0]
            if coord[3] < coord[1]:
                coord[1], coord[3] = coord[3], coord[1]
            road = roads_repair(UPLOAD_FOLDER + '/result.png', UPLOAD_FOLDER + '/score.npy', coord)
            ori = cv.imread(UPLOAD_FOLDER + '/origin.png')
            ori_o = ori.copy()
            ori_c = clip_img(ori, coord)
            ori_oc = ori_c.copy()
            ori_c[road == 255] = [1, 0, 255]
            comp_prev = oe_mix(ori_oc, ori_c)
            cv.imwrite(UPLOAD_FOLDER + '/clipP.png', comp_prev)

            road_o = cv.imread(UPLOAD_FOLDER + '/result.png')
            lut = np.zeros((256, 3), dtype=np.uint8)
            lut[255] = [255, 255, 255]
            lut[0] = [0, 0, 0]
            road = lut[road]
            road_o[coord[1]:coord[3], coord[0]:coord[2]] = road

            ori[road_o[:, :, 0] == 255] = [1, 0, 255]
            comp = oe_mix(ori_o, ori)
            cv.imwrite(UPLOAD_FOLDER + '/mixed.png', comp)

            cv.imwrite(UPLOAD_FOLDER + '/result.png', road_o)
            road_o[road_o[:, :, 0] == 255] = [1, 0, 255]
            road_o = add_alpha(road_o)
            cv.imwrite(UPLOAD_FOLDER + '/roads.png', road_o)

            return {'h': comp_prev.shape[0], 'w': comp_prev.shape[1]}
        elif status == 'move':
            no = request.values['no']
            if pro == '0':
                shutil.copy(UPLOAD_FOLDER + '/multi/' + no + '/origin.png', UPLOAD_FOLDER)
                shutil.copy(UPLOAD_FOLDER + '/multi/' + no + '/result.png', UPLOAD_FOLDER)
                shutil.copy(UPLOAD_FOLDER + '/multi/' + no + '/roads.png', UPLOAD_FOLDER)
                shutil.copy(UPLOAD_FOLDER + '/multi/' + no + '/mixed.png', UPLOAD_FOLDER)
            elif pro == '2':
                shutil.copy(UPLOAD_FOLDER + '/multi/' + no + '/origin.png', UPLOAD_FOLDER)
                shutil.copy(UPLOAD_FOLDER + '/multi/' + no + '/result.png', UPLOAD_FOLDER)
            elif pro == '3':
                shutil.copy(UPLOAD_FOLDER + '/multi/' + no + '/origin.png', UPLOAD_FOLDER)
                shutil.copy(UPLOAD_FOLDER + '/multi/' + no + '/result.png', UPLOAD_FOLDER)
                shutil.copy(UPLOAD_FOLDER + '/multi/' + no + '/mixed.png', UPLOAD_FOLDER)
                for k in range(4):
                    shutil.copy(UPLOAD_FOLDER + '/multi/' + no + '/class' + str(k) + '.png', UPLOAD_FOLDER)
            return 'DONE'


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    app.run(port=8080)
