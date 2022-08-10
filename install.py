import shutil
import logging
import warnings
import os

warnings.filterwarnings("ignore", category=Warning)
logging.basicConfig(
    level=logging.INFO,  # 定义输出到文件的log级别，
    format='[%(asctime)s - %(levelname)s - %(filename)s : %(funcName)s ] - %(message)s',  # 定义输出log的格式
    datefmt='%Y-%m-%d %A %H:%M:%S',  # 时间
)


def deploy():
    logging.info('开始解压')
    shutil.unpack_archive('./PaddleRS.zip', './PaddleRS/')
    shutil.unpack_archive('./data_demo.zip', './data_demo/')
    logging.info('解压完成')
    logging.info('安装PaddleRS')
    os.system('pip install ./PaddleRS')
    logging.info('安装完毕')


if __name__ == '__main__':
    deploy()
