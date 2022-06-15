import math
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from functions.object_detection import load_object_detection, object_detection
from functions.object_classification import load_object_classification, object_classification
import cv2 as cv

UPLOAD_FOLDER = './webpage/res'
WEIGHT_FOLDER = './weights'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODELS = ['playground', 'aircraft', 'oiltank', 'overpass']

app = Flask(__name__, template_folder="./webpage", static_folder='./webpage', static_url_path="")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
pro, model, loc, score, period, shape = [0] * 6


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
        if pro == '2':
            pro_type = int(request.values['type'])
            model = load_object_detection(WEIGHT_FOLDER + '/object_detection/' + MODELS[pro_type])
        elif pro == '3':
            model = load_object_classification(WEIGHT_FOLDER + '/object_classification')
        return redirect(url_for('upload_file'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
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
                filename1 = secure_filename(file1.filename)
                file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
                filename2 = secure_filename(file2.filename)
                file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
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
                global loc, score, period, shape
                if pro == '2':
                    res, loc, score, period = object_detection(save_path, model)
                    shape = list(res.shape)
                    cv.imwrite(UPLOAD_FOLDER + '/result.png', res)
                if pro == '3':
                    res, period = object_classification(save_path, model)
                    cv.imwrite(UPLOAD_FOLDER + '/result.png', res)
                return redirect(url_for('main_process'))


@app.route('/result', methods=['GET', 'POST'])
def main_process():
    if request.method == 'GET':
        if pro == '2':
            return render_template('main.html', pro=pro, ele=[loc, score, period, shape])
        elif pro == '3':
            return render_template('main.html', pro=pro, ele=[period])
        else:
            return render_template('main.html', pro=pro)
    elif request.method == 'POST':
        status = request.values['status']
        if status == 'clip':
            coord = [request.values['x0'], request.values['y0'], request.values['x1'], request.values['y1']]
            coord = [math.floor(float(k)) for k in coord]
            img = cv.imread(UPLOAD_FOLDER + '/origin.png')
            clipped = clip_img(img, coord)
            cv.imwrite(UPLOAD_FOLDER + '/clip.png', clipped)
            return {'h': clipped.shape[0], 'w': clipped.shape[1]}


if __name__ == '__main__':
    app.run(port=8080, debug=True)
