#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
from config import cfg
from pprint import pprint
from datetime import datetime as dt
from argparse import ArgumentParser

import logging
import matplotlib
import multiprocessing as mp
import numpy as np
import os
import sys
import tensorflow as tf

import utils.data_loaders
from model import Pix2VoxModel

# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')


def parse_args():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    # parser.add_argument('--gpu',
    #                     dest='gpu_id',
    #                     help='GPU device id to use [cuda0]',
    #                     default=cfg.CONST.DEVICE,
    #                     type=str)
    # parser.add_argument('--rand', dest='randomize',
    #                     help='Randomize (do not use a fixed seed)', action='store_true')
    # parser.add_argument('--test', dest='test',
    #                     help='Test neural networks', action='store_true')
    parser.add_argument('--task', dest='task',
                        choices=['train', 'test', 'both'],
                        help='training, testing, or both',
                        default=cfg.TASK.TASK_TYPE)
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='name of the net',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--epoch', dest='epoch', help='number of epoches',
                        default=cfg.TRAIN.NUM_EPOCHES, type=int)
    parser.add_argument('--weights', dest='weights',
                        help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path',
                        help='Set output path', default=cfg.DIR.OUT_PATH)
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = parse_args()

    # if args.gpu_id is not None:
    #     cfg.CONST.DEVICE = args.gpu_id
    # if not args.randomize:
    #     np.random.seed(cfg.CONST.RNG_SEED)
    if args.task is not None:
        cfg.TASK.TASK_TYPE = args.task
    if args.batch_size is not None:
        cfg.CONST.BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        cfg.TRAIN.NUM_EPOCHES = args.epoch
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True

    # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    # if type(cfg.CONST.DEVICE) == str:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    ######################## DATA LOADING ########################
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](
        cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](
        cfg)

    train_dataset = train_dataset_loader.load_dataset_files(
        'train', cfg.CONST.N_VIEWS_RENDERING)
    val_dataset = val_dataset_loader.load_dataset_files(
        'val', cfg.CONST.N_VIEWS_RENDERING)

    train_data = (train_dataset.images, train_dataset.vols,
                  train_dataset.taxonomy_names, train_dataset.sample_names)
    val_data = (val_dataset.images, val_dataset.vols,
                val_dataset.taxonomy_names, val_dataset.sample_names)

    ########################## TRAINING ##########################
    model = Pix2VoxModel(cfg)

    if cfg.TASK.TASK_TYPE in ['train', 'both']:
        model.compile(tf.keras.optimizers.Adam(),
                      tf.keras.losses.BinaryCrossentropy())
        model.train(train_data, val_data)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception(
            "Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    # Setup logger
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    main()
