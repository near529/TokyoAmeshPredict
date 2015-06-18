#!/usr/bin/env python
# -*- coding: utf-8 -*-
u'''Predict module'''

from skimage import io
import numpy as np
from matplotlib import pyplot as plt

from utils import ImageBase, Grid
from settings import Settings
from regist import RegularRegist

def _test():
    moving_path = r'../TOKYO_AMESH_IMAGE/000/2009/200905290240.gif'
    fixed_path = r'../TOKYO_AMESH_IMAGE/000/2009/200905290250.gif'
    real_path = r'../TOKYO_AMESH_IMAGE/000/2009/200905290300.gif'

    moving = ImageBase()
    moving.load_data(moving_path)

    
    fixed = ImageBase()
    fixed.load_data(fixed_path)
    regist = RegularRegist(moving, fixed)
    regist.run()
    regist.predict_linear_transform()
    
    # For display
    # regist.regist_linear_transform()
    # img_moved = convert_to_RGB(regist.vmoved.values)
    
    img_next = convert_to_RGB(regist.vnext.values)
    img_real = io.imread(real_path)
    img_moving = io.imread(moving_path)
    img_fixed = io.imread(fixed_path)

    display(img_moving, img_fixed, img_real, img_next)
    #print regist.optimizer.vgrid.values
    #print regist.fixed.values
    #print regist.optimizer.cost

def convert_to_RGB(vmatrix):
    u'''Convert moved image to RGB image'''
    (nrow, ncol) = vmatrix.shape
    dt_stone = Settings.DATA_STONE
    dt_bound = Settings.DATA_BOUNDER
    bound_num = len(dt_bound)
    dt_max = Settings.DATA_MAX
    img = np.zeros((nrow, ncol, 3), dtype=np.uint8)
    for i in range(nrow):
        for j in range(ncol):
            value = vmatrix[i, j] * dt_max
            idx = bound_num-1
            while idx > 0 and dt_bound[idx] > value:
                idx -= 1
            img[i, j] = dt_stone[idx]
    return img

def display(img_moving, img_fixed, img_real, img_next):
    u'''Display moving, fixed, moved and next'''
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(img_moving, cmap='gray')
    axes[0, 0].set_title('Moving')

    axes[0, 1].imshow(img_fixed, cmap='gray')
    axes[0, 1].set_title('Fixed')

    axes[1, 0].imshow(img_real, cmap='gray')
    axes[1, 0].set_title('Real')

    axes[1, 1].imshow(img_next, cmap='gray')
    axes[1, 1].set_title('Predicted')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
    fig.suptitle('Amesh prediction')
    plt.show()

def test_values_img():
    u'''Test value in images'''
    sample = set()
    for year in range(2009, 2010):
        folder_dir = os.path.join(Settings.DATA_DIR, str(year))
        for filepath in glob.glob(folder_dir + '/*.gif'):
            imgb = ImageBase()
            sample = sample.union(imgb.get_data(filepath))
            print filepath, sample

if __name__ == '__main__':
    _test()
