conda create -n I2RSI python=3.8 --y && conda activate I2RSI && conda install swig --yes && python -m pip install paddlepaddle==2.3.0 -i https://mirror.baidu.com/pypi/simple && pip install Flask pywebview filelock protobuf==3.20 && python ./install.py && echo "DONE" && pause