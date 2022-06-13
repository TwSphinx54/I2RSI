import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './imgs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder="./webpage", static_folder='./webpage', static_url_path="")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
pro = 0


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    if request.method == 'GET':
        return render_template('welcome.html')
    elif request.method == 'POST':
        global pro
        pro = request.values['pro']
        return redirect(url_for('upload_file'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html', pro=pro)
    elif request.method == 'POST':
        if pro == 1:
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
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('main_process'))


@app.route('/result', methods=['GET', 'POST'])
def main_process():
    return render_template('main.html', pro=pro)


if __name__ == '__main__':
    app.run(port=8080, debug=True)
