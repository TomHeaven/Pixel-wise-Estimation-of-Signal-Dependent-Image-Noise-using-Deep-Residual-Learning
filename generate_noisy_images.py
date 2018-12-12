# coding=utf8
# Author: TomHeaven, hanlin_tan@nudt.edu.cn, 2018.06.27

from __future__ import print_function

import cv2
import re
import os
import numpy as np
import math


def add_noise(input_folder, output_folder, sigma, is_homo=True):
    regexp = re.compile(r'.*\.(%s)' % '(jpg)|(png)|(bmp)|(tif)')

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    n = 0
    for d, dirs, files in os.walk(input_folder):
        for f in files:
            if regexp.match(f):
                print('image', n,  f)

                image = cv2.imread(os.path.join(d, f))
                image = np.float32(image)

                if is_homo:
                    s = sigma
                else:
                    s = (np.random.rand(1) - 0.5) * 5 + sigma
                noisy = image + np.random.randn(image.shape[0], image.shape[1], image.shape[2]) * s
                # clip is very important
                noisy = np.uint8(np.clip(noisy, 0, 255))

                if f.endswith('.jpg'):
                    f = f.replace('.jpg', '.png')

                cv2.imwrite(os.path.join(output_folder, f), noisy)
                n += 1

def add_noise_v2(input_folder, output_folder, sigma, min_noise_level=0, max_noise_level=30):
    regexp = re.compile(r'.*\.(%s)' % '(jpg)|(png)|(bmp)|(tif)')

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    n = 0
    for d, dirs, files in os.walk(input_folder):
        for f in files:
            if regexp.match(f):
                print('image', n,  f)

                image = cv2.imread(os.path.join(d, f))
                image = np.float32(image)


                s = np.random.rand(1)  * (max_noise_level - min_noise_level) + min_noise_level
                noisy = image + np.random.randn(image.shape[0], image.shape[1], image.shape[2]) * s
                # clip is very important
                noisy = np.uint8(np.clip(noisy, 0, 255))

                if f.endswith('.jpg'):
                    f = f.replace('.jpg', '.png')

                cv2.imwrite(os.path.join(output_folder, f), noisy)
                n += 1

def add_noise_v3(input_folder, output_folder, sigma, min_noise_level=0, max_noise_level=15):
    regexp = re.compile(r'.*\.(%s)' % '(jpg)|(png)|(bmp)|(tif)')

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    n = 0

    for d, dirs, files in os.walk(input_folder):
        for f in files:
            if regexp.match(f):
                print('image', n,  f)

                image = cv2.imread(os.path.join(d, f))
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                image = np.float32(image)
                #gray = np.float32(gray)

                sigma_s = np.random.rand(1)  * 3.5
                sigma_r = np.random.rand(1)  * (max_noise_level - min_noise_level) + min_noise_level

                s = np.sqrt(sigma_s**2 + sigma_r * image)
                # Prevent s from exceeding prediction range of DRNE
                s = np.clip(s, 0, 30)
                noisy = image + np.random.randn(image.shape[0], image.shape[1], image.shape[2]) * s
                # clip is very important
                noisy = np.uint8(np.clip(noisy, 0, 255))

                if f.endswith('.jpg'):
                    f = f.replace('.jpg', '.png')

                cv2.imwrite(os.path.join(output_folder, f), noisy)
                n += 1

# 分段非均匀
def generate_noisy_datasets(is_homo):
    add_noise('data/mcm', 'data/mcm_sigma5', 5, is_homo)
    add_noise('data/mcm', 'data/mcm_sigma15', 15, is_homo)
    add_noise('data/mcm', 'data/mcm_sigma25', 25, is_homo)

    add_noise('data/kodak', 'data/kodak_sigma5', 5, is_homo)
    add_noise('data/kodak', 'data/kodak_sigma15', 15, is_homo)
    #add_noise('data/kodak', 'data/kodak_sigma25', 25, is_homo)

    add_noise('data/bsd500', 'data/bsd500_sigma5', 5, is_homo)
    add_noise('data/bsd500', 'data/bsd500_sigma15', 15, is_homo)
    add_noise('data/bsd500', 'data/bsd500_sigma25', 25, is_homo)

# 0-30 非均匀
def generate_noisy_datasets_v2():
    add_noise_v2('data/mcm', 'data/mcm_sigma15', 15)
    add_noise_v2('data/kodak', 'data/kodak_sigma15', 15)
    add_noise_v2('data/bsd500', 'data/bsd500_sigma15', 15)

# Real noise model 非均匀
def generate_noisy_datasets_v3():
    add_noise_v3('data/mcm', 'data/mcm_sigma15', 15)
    add_noise_v3('data/kodak', 'data/kodak_sigma15', 15)
    add_noise_v3('data/bsd500', 'data/bsd500_sigma15', 15)



if __name__ == '__main__':
    #######
    # Use only one of the following functions at one time
    #######

    # to generate homogeneous noise
    #generate_noisy_datasets(is_homo=True)

    # to generate non-homogeneous noise in three ranges: [0, 9], [10, 19], [20, 29]
    #generate_noisy_datasets(is_homo=False)

    # to generate non-homogeneous noise in range: [0, 29]
    #generate_noisy_datasets_v2()

    # to genereate  non-homogeneous using noise model (1)
    generate_noisy_datasets_v3()
