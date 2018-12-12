# coding=utf8
# Author: TomHeaven, hanlin_tan@nudt.edu.cn, 2018.06.27

from __future__ import print_function
import cv2
import re
import os

import numpy as np
import math


def get_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compare_results(folder1, folder2):
    regexp = re.compile(r'.*\.(%s)' % '(jpg)|(png)|(bmp)|(tif)')

    n = 0

    psnr_list = []
    for d, dirs, files in os.walk(folder1):
        for f in files:
            if regexp.match(f):
                image1 = cv2.imread(os.path.join(folder1, f))

                # for bsd500 (.jpg)
                if f.endswith('.jpg'):
                    f = f.replace('.jpg', '.png')
                image2 = cv2.imread(os.path.join(folder2, f))
                psnr = get_psnr(image1, image2)
                psnr_list.append(psnr)

    psnr_array = np.asarray(psnr_list)
    avg_psnr = psnr_array.mean()
    return avg_psnr, psnr_array

def compare_all():
    methods = ['drne+dn', 'chen+dn']
    #datasets = ['kodak', 'mcm', 'bsd500']
    #methods = ['chen+dn']
    datasets = ['kodak']
    #sigmas = [5, 15, 25]
    sigmas = [15]


    for m in methods:
        print('Method : ', m)
        for d in datasets:
            print('  Dataset : ', d)

            folder1 = 'data/%s' % d
            for s in sigmas:
                print('    Sigma : ', s)

                folder2 = 'res/%s/%s_sigma%d' % (m, d, s)

                avg_psnr, psnr_array = compare_results(folder1, folder2)
                print('      average psnr : ', avg_psnr)

if __name__ == '__main__':
    compare_all()
