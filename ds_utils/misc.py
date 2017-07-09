import os
import datetime
import numpy as np
import glob
import shutil

import matplotlib.pyplot as plt
import IPython.display

import keras.utils.vis_utils

from inspect import getsourcelines

def move_sample(dir_source, dir_destin, file_type, n):
    """

    :param dir_source:
    :param dir_destin:
    :param file_type:
    :param n:
    :return:

    Example:
        move_sample(dir_source=data_dir+'/preprocessed/train', dir_destin=data_dir+'/preprocessed/sample/train', file_type='jpg', 200)

    """
    g = glob.glob(dir_source + '/*' + file_type)
    fs = [pth.split('/')[-1] for pth in g]
    fs_shuffle = np.random.permutation(fs)
    for i in range(n):
        os.rename(dir_source + '/' + fs_shuffle[i], dir_destin + '/' + fs_shuffle[i])


def copy_sample(dir_source, dir_destin, file_type, n):
    """

    :param dir_source:
    :param dir_destin:
    :param file_type:
    :param n:
    :return:

    Example:
        copy_sample(dir_source=data_dir+'/preprocessed/train', dir_destin=data_dir+'/preprocessed/sample/train', file_type='jpg', 200)

    """
    g = glob.glob(dir_source + '/*' + file_type)
    fs = [pth.split('/')[-1] for pth in g]
    fs_shuffle = np.random.permutation(fs)
    for i in range(n):
        shutil.copyfile(dir_source + '/' + fs_shuffle[i], dir_destin + '/' + fs_shuffle[i])


def imshow_gray(array):
    plt.imshow(array, cmap=plt.cm.gray_r)


def datetime_now_string():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def plot_keras_model(model, show_shapes=True):
    return IPython.display.SVG(keras.utils.vis_utils.model_to_dot(model, show_shapes=show_shapes).create(prog='dot', format='svg'))


def print_source(obj):
    print(''.join(getsourcelines(obj)[0]))

