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

from model_noise_estimation_w64d16_v2_sigma0_30 import Estimator
from drdd_dn_sigma0_50 import DeepProcesser



def test_estimate(input_folder, modelPath, feature_dim, depth, device):
    """
    Denoise noisy images using Estimator class with pre-trained model.
    :param modelPath: path to save trained model
    :param feature_dim: width of the DNN
    :param depth: depth of the DNN
    :param device: which GPU to use (for machines with multiple GPUs, this avoid taking up all GPUs)
    :param noise: standard variation of noise of the tested images
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    estimator = Estimator(batchSize=1, feature_dim=feature_dim, depth=depth)
    regexp = re.compile(r'.*\.(%s)' % '(jpg)|(png)|(bmp)|(tif)')
    #input_folder = 'data/mcm'

    psize = 250
    max_value = 255.0

    crop = 0
    n = 0

    avg_en = 0

    outFile = h5py.File('data/ne_res.h5', "w")

    for d, dirs, files in os.walk(input_folder):
        for f in files:
            if regexp.match(f):
                print('image', n,  f)

                image = cv2.imread(os.path.join(d, f))
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                if DEBUG:
                    print ('image.shape : ', image.shape)

                if n == 0:
                    xshape = [psize, psize, 3]
                    yshape = [psize, psize, 3]
                    estimator.load_model(modelPath, batchSize=1, xshape = xshape, yshape=yshape)


                image = image / max_value
                R, runtime = estimator.denoise_bayer(image, psize, crop)
                estimated_noise = np.mean(np.mean(np.mean(R, axis=0), axis=0), axis=0)

                if DEBUG:
                    print('max value = ', np.max(np.abs(R)))
                    print('time : ', runtime, ' ms')

                outFile.create_dataset('noise_estimation_%d' % n, data=np.mean(R, axis=2), compression='gzip')
                #outFile.create_dataset('runtime_%d' % n, data=R * 255, compression='gzip')
                print('estimate_noise : ', estimated_noise * 255.0)
                n += 1
                avg_en += estimated_noise

    outFile.close()
    print('avg_en : ', avg_en / n * 255.0)
    estimator.sess.close()

def test_denoise(input_folder, output_folder, model_dir, block_num, width, block_depth, device):
    ## denoise
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    deepProcesser = DeepProcesser(block_num=block_num, width=width, block_depth=block_depth,
                                  use_scalar_noise=False)
    deepProcesser.load_model(model_dir, False)

    regexp = re.compile(r'.*\.(%s)' % '(tif|tiff|jpg|png)')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    psize = 250
    max_value = 255.0

    crop = 0
    n = 0

    nl_file = h5py.File('data/ne_res.h5', "r")

    for d, dirs, files in os.walk(input_folder):
        for f in files:
            if regexp.match(f):
                print('image', n, f)
                image = cv2.imread(os.path.join(d, f))

                if DEBUG:
                    print('image.shape : ', image.shape)

                image = image / max_value

                noise = nl_file['noise_estimation_%d' % n].value
                #print('noise.shape : ',  noise.shape)
                #noise = noise.transpose(2, 0, 1)
                noise = noise[np.newaxis, np.newaxis, ...]

                R, runtime = deepProcesser.test(image, noise, psize, crop)
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

    datasets = ['kodak', 'mcm', 'bsd500']
    #sigmas = [5, 15, 25]
    sigmas = [15]


    for d in datasets:
        print('  Dataset : ', d)
        for s in sigmas:
            print('    Sigma : ', s)

            input_folder = 'data/%s_sigma%d' % (d, s)
            output_folder= 'res/drne+dn/%s_sigma%d' % (d, s)
            # estimation
            modelPath = 'ne_w64d16_v2_sigma0_30'
            width = 64
            depth = 16 - 4
            minNoiseLevel = 0.0 / 255.0
            maxNoiseLevel = 30.0 / 255.0
            #### end configuration

            # 用子进程启动 Tensorflow，退出时可完全释放显存
            p = Process(target=test_estimate, args=(input_folder, 'models/%s' % modelPath, width, depth, device))
            p.start()
            p.join()  # this blocks until the process terminates
            #test_estimate(input_folder, 'models/%s' % modelPath, width, depth=depth, device=device)

            # denoise
            modelPath = 'dn_sigma0_50'
            block_num = 5
            block_depth = 4
            width = 64
            # 用子进程启动 Pytorch，退出时可完全释放显存
            p = Process(target=test_denoise, args=(input_folder,  output_folder, 'models/%s' % modelPath, block_num, width, block_depth, device))
            p.start()
            p.join()  # this blocks until the process terminates
            #test_denoise(input_folder,  output_folder, 'models/%s' % modelPath, block_num, width, block_depth, device=device)