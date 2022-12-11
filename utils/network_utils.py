import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

import utils.binvox_rw
from utils.binvox_rw import Voxels
from datetime import datetime as dt


def save_volume(volume, save_dir, n_itr):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    volume = volume.squeeze()
    volume = volume.__ge__(3e-2)
    dims = [32, 32, 32]
    translation = [-0.358972, -0.0937029, -0.335199]
    scale = 0.75
    axis_order = 'xyz'

    save_file = os.path.join(save_dir, 'voxels-%06d.binvox' % (n_itr + 1))

    with open(save_file, 'wb') as fp:
        utils.binvox_rw.write(Voxels(
            volume, dims, translation, scale, axis_order), fp)

    print('[INFO] %s Saving file to %s' % (dt.now(), save_file))


# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

class AverageMeter(object):
    '''
    Class that calculates the average of a particular metric.
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
