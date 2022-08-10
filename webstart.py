import sys
import threading
import webview
from process import app


def start_sever():
    app.run(port=8080)


if __name__ == '__main__':
    t = threading.Thread(target=start_sever)
    t.daemon = True
    t.start()
    webview.create_window('I2RSI遥感图像智能解译系统', url='http://127.0.0.1:8080/welcome', width=1920, height=1080)
    webview.start()
    sys.exit()
