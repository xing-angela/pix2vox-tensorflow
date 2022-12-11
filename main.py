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
from tensorboardX import SummaryWriter

# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')


def parse_args():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
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

    ######################## DATA LOADING ########################
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](
        cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](
        cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](
        cfg)

    train_dataset = train_dataset_loader.load_dataset_files(
        'train', cfg.CONST.N_VIEWS_RENDERING)
    val_dataset = val_dataset_loader.load_dataset_files(
        'val', cfg.CONST.N_VIEWS_RENDERING)
    test_dataset = test_dataset_loader.load_dataset_files(
        'test', cfg.CONST.N_VIEWS_RENDERING)

    train_data = (train_dataset.images, train_dataset.vols,
                  train_dataset.taxonomy_names, train_dataset.sample_names)
    val_data = (val_dataset.images, val_dataset.vols,
                val_dataset.taxonomy_names, val_dataset.sample_names)
    test_data = (test_dataset.images, test_dataset.vols,
                 test_dataset.taxonomy_names, test_dataset.sample_names)

    ########################## TRAINING ##########################
    model = Pix2VoxModel(cfg)

    model.compile(
        tf.keras.optimizers.Adam(
            learning_rate=cfg.TRAIN.LEARNING_RATE,
            beta_1=cfg.TRAIN.BETAS[0],
            beta_2=cfg.TRAIN.BETAS[1]),
        tf.keras.losses.BinaryCrossentropy())

    # trains the model
    model.train(train_data, val_data)

    # tests the model
    output_dir = os.path.join(
        cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    log_dir = output_dir % 'logs'
    test_writer = SummaryWriter(os.path.join(log_dir, 'evaluation'))
    model.test(test_data, -1, output_dir, test_writer)


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
