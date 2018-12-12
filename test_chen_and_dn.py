# coding=utf8
# Author: TomHeaven, hanlin_tan@nudt.edu.cn, 2017.08.19

from __future__ import print_function
from tensorflow.contrib.layers import conv2d, avg_pool2d
import tensorflow as tf
import numpy as np
from data_v3 import DatabaseCreator
import time
import tqdm
import cv2
import re
import os
import argparse
import h5py
from multiprocessing import Process

# options
DEBUG = False

from drdd_dn_sigma0_50 import DeepProcesser



def test_denoise(input_folder, sigma_folder, output_folder, model_dir, block_num, width, block_depth, device):
    ## denoise
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    deepProcesser = DeepProcesser(block_num=block_num, width=width, block_depth=block_depth,
                                  use_scalar_noise=True)
    deepProcesser.load_model(model_dir, False)

    regexp = re.compile(r'.*\.(%s)' % '(tif|tiff|jpg|png)')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    psize = 250
    max_value = 255.0

    crop = 0
    n = 0

    for d, dirs, files in os.walk(input_folder):
        for f in files:
            if regexp.match(f):
                print('image', n, f)
                image = cv2.imread(os.path.join(d, f))

                if DEBUG:
                    print('image.shape : ', image.shape)

                image = image / max_value
                with open('%s/%s_chen.txt' % (sigma_folder, f), 'r') as sigma_file:
                    str = sigma_file.readline()
                    noise_level = np.float32(str)
                    print('noise_level : ', noise_level)

                R, runtime = deepProcesser.test(image, noise_level / 255.0, psize, crop)
                out = np.uint8(R * 255 + 0.5)

                # R = swap_blue_red(R)

                if DEBUG:
                    print('max value = ', np.max(np.abs(R)))
                    print('time : ', runtime, ' ms')

                cv2.imwrite(os.path.join(output_folder, f), out)
                n += 1

if __name__ == '__main__':
    ## configuration
    device = '1'
    # datasets: 'kodak', 'mcm', 'bsd500' sigma 5, 15, 25
    methods = ['drne+dn', 'chen+dn']
    datasets = ['kodak', 'mcm', 'bsd500']
    #sigmas = [5, 15, 25]
    sigmas = [15]


    for d in datasets:
        print('  Dataset : ', d)
        for s in sigmas:
            print('    Sigma : ', s)

            input_folder = 'data/%s_sigma%d' % (d, s)
            sigma_folder = 'data/chen/%s_sigma%d' %(d, s)
            output_folder= 'res/chen+dn/%s_sigma%d'  %(d, s)
            #### end configuration

            # denoise
            modelPath = 'dn_sigma0_50'
            block_num = 5
            block_depth = 4
            width = 64

            # 用子进程启动 Pytorch，退出时可完全释放显存
            p = Process(target=test_denoise, args=(
            input_folder, sigma_folder, output_folder, 'models/%s' % modelPath, block_num, width, block_depth, device))
            p.start()
            p.join()  # this blocks until the process terminates
            #test_denoise(input_folder, sigma_folder, output_folder, 'models/%s' % modelPath, block_num, width, block_depth, device=device)