import math
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from functions.object_detection import load_object_detection, object_detection
from functions.object_classification import load_object_classification, object_classification
from functions.object_extraction import load_object_extraction, object_extraction
from functions.change_detection import load_change_detection, change_detection
import cv2 as cv

UPLOAD_FOLDER = './webpage/res'
WEIGHT_FOLDER = './weights'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODELS = ['playground', 'aircraft', 'oiltank', 'overpass']

app = Flask(__name__, template_folder="./webpage", static_folder='./webpage', static_url_path="")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
pro, model, loc, score, period, shape, areas = [0] * 7


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clip_img(img, coord):
    return img[coord[1]:coord[3], coord[0]:coord[2]]


@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    if request.method == 'GET':
        return render_template('welcome.html')
    elif request.method == 'POST':
        global pro, model
        pro = request.values['pro']
        if pro == '0':
            model = load_object_extraction(WEIGHT_FOLDER + '/object_extraction/')
        elif pro == '1':
            model = load_change_detection(WEIGHT_FOLDER + '/change_detection/')
        elif pro == '2':
            pro_type = int(request.values['type'])
            model = load_object_detection(WEIGHT_FOLDER + '/object_detection/' + MODELS[pro_type])
        elif pro == '3':
            model = load_object_classification(WEIGHT_FOLDER + '/object_classification')
        return redirect(url_for('upload_file'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global loc, score, period, shape, areas
    if request.method == 'GET':
        return render_template('index.html', pro=pro)
    elif request.method == 'POST':
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
                res, alpha, score, period = change_detection(A, B, model)
                shape = list(res.shape)
                cv.imwrite(UPLOAD_FOLDER + '/result.png', res)
                cv.imwrite(UPLOAD_FOLDER + '/change.png', alpha)
                print(period)
                return redirect(url_for('main_process'))
        else:
            file = request.files['image']
            # If the user does not select a file, the browser submits an empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                # filename = secure_filename(file.filename)
                filename = 'origin.png'
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                if pro == '0':
                    res, roads, score, period, mixed = object_extraction(save_path, model)
                    shape = list(res.shape)
                    cv.imwrite(UPLOAD_FOLDER + '/result.png', res)
                    cv.imwrite(UPLOAD_FOLDER + '/roads.png', roads)
                    cv.imwrite(UPLOAD_FOLDER + '/mixed.png', mixed)
                elif pro == '2':
                    res, loc, score, period = object_detection(save_path, model)
                    shape = list(res.shape)
                    cv.imwrite(UPLOAD_FOLDER + '/result.png', res)
                elif pro == '3':
                    res, types, score, period, areas = object_classification(save_path, model)
                    cv.imwrite(UPLOAD_FOLDER + '/result.png', res)
                    shape = list(res.shape)
                    for k in range(4):
                        cv.imwrite(UPLOAD_FOLDER + '/class' + str(k) + '.png', types[k])
                return redirect(url_for('main_process'))


@app.route('/result', methods=['GET', 'POST'])
def main_process():
    if request.method == 'GET':
        if pro == '1':
            return render_template('main.html', pro=pro, ele=[score, period, shape])
        elif pro == '2':
            return render_template('main.html', pro=pro, ele=[loc, score, period, shape])
        elif pro == '3':
            return render_template('main.html', pro=pro, ele=[score, period, shape, areas])
        else:
            return render_template('main.html', pro=pro, ele=[score, period, shape])
    elif request.method == 'POST':
        status = request.values['status']
        if status == 'clip':
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


if __name__ == '__main__':
    app.run(port=8080, debug=True)
