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

# options
DEBUG = False

class Estimator:
    """
    A class to train and test a tensorflow estimator.
    """

    # predict_op = []
    def __init__(self, batchSize = 32, depth = 8, feature_dim = 8, device = '/gpu:0', xshape=[128,128,3], yshape=[128,128,3], lr=1e-4):
        self.batchSize = batchSize
        self.depth = depth
        self.feature_dim = feature_dim
        self.device = device
        self.xshape = xshape
        self.yshape = yshape
        self.lr = lr

    def init_weights(self, shape, name):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    def residual_block(self, h, width, kernel_size, depth):
        h_in = h
        for i in range(depth):
            h = conv2d(h, width, kernel_size)
        return h_in + h

    def build_model(self, bTrain):
        assert len(self.xshape) == 3
        lmd = 0.25
        # place holders
        x = tf.placeholder('float', [self.batchSize, self.xshape[0], self.xshape[1], self.xshape[2]], 'x')

        if bTrain:
            noise_level = tf.placeholder('float', shape=(1), name='noise')
            noise = tf.fill([self.batchSize, self.xshape[0], self.xshape[1], 1], noise_level[0])
            # y = x
            gaussian_noise = tf.random_normal(shape=tf.shape(x), stddev=noise_level[0], dtype=tf.float32)
            h = x + gaussian_noise
        else:
            h = x

        # start data flow
        block_depth = 4
        num_block = self.depth / block_depth
        for d in range(0, num_block):
            #h = tf.concat([h, noise], axis=3, name='concat_%d' % d)
            h = conv2d(h, self.feature_dim, [3, 3])
            h = self.residual_block(h, self.feature_dim, [3, 3], block_depth)

        h = conv2d(h, 1 , [3, 3])

        y_conv = h
        scalar_en = tf.reduce_mean(h)

        # loss function
        if bTrain:
            #cost_mat = tf.reduce_mean(tf.square(tf.subtract(noise,  y_conv))) * self.batchSize
            cost_mat = tf.reduce_sum(tf.square(tf.subtract(noise,  y_conv))) / self.batchSize
            cost_scalar = tf.square(tf.subtract(scalar_en, noise_level[0]))

            cost = lmd * cost_mat + (1 - lmd) * cost_scalar
            #cost = tf.nn.l2_loss(y - y_conv)
            train_op = tf.train.AdamOptimizer(self.lr).minimize(cost)
            #train_op = tf.train.GradientDescentOptimizer(1e-4)(cost)
            return y_conv, train_op, cost, x, noise_level
        else:
            return y_conv, x

    def train(self, saveDir,  trY,  valY, minNoiseLevel, maxNoiseLevel, maxEpoch=1000, part=0):
        """
        train
        :param trX:
        :param trY:
        :param maxEpoch:
        :param batchSize:
        :return:
        """

        # add new axis for data
        if trY.ndim == 3:
            trY = trY[..., np.newaxis]
        if valY.ndim == 3:
            valY = valY[..., np.newaxis]

        # generate model
        if not hasattr(self, 'predict_op'):
            print('Building model ...')
            self.predict_op, self.train_op, self.cost, self.x, self.noise_level = self.build_model(bTrain=True)
        # Launch the graph in a session
        saver = tf.train.Saver()

        if not os.path.isdir(saveDir):
            os.mkdir(saveDir)

        curEpoch = 0
        bestLoss = 99999.0
        if os.path.isfile(saveDir + '/loss.txt'):
            with open(saveDir + '/loss.txt', 'r') as log_file:
                log = log_file.readlines()
                if len(log) > 0:
                    curEpoch = int(log[-1].split(' ')[0]) + 1 + part * maxEpoch

        out_file = open(saveDir + '/loss.txt', 'a')
        with tf.Session() as sess:
            self.sess = sess

            with tf.device(self.device):
                ckpt = tf.train.get_checkpoint_state(saveDir)
                if ckpt and ckpt.model_checkpoint_path:
                    print('Restored training...')
                    saver.restore(sess, saveDir + '/tf_estimator.ckpt')
                else:
                    print('Start training...')
                    # init all variables
                    tf.global_variables_initializer().run()

                for i in range(curEpoch, maxEpoch):
                    start_time = time.time()
                    print('Epoch %d ...' % i)
                    for start, end in zip(range(0, len(trY), self.batchSize),
                                          range(self.batchSize, len(trY) + 1, self.batchSize)):

                        y = trY[start:end]

                        n_level = np.random.rand(1) * (maxNoiseLevel - minNoiseLevel) + minNoiseLevel

                        sess.run(self.train_op, feed_dict={self.x: y, self.noise_level: n_level})

                    # print loss
                    for n_level in [5, 15, 25]:
                        loss = sess.run(self.cost, feed_dict={self.x: trY[:self.batchSize, ...],
                                                             self.noise_level: [n_level / 255.0]})
                        val_loss = sess.run(self.cost, feed_dict={self.x: valY[:self.batchSize, ...],
                                                                 self.noise_level: [n_level / 255.0]})
                        print('loss n : ', n_level, loss, ' val loss : ', val_loss)
                        print(i, n_level, loss, val_loss, file=out_file)
                    print('time : ', time.time() - start_time, ' s')


                    if i % 10 == 0:
                        if val_loss < bestLoss or i < maxEpoch * 4 / 5:
                            bestLoss = val_loss
                            saver.save(sess, saveDir + '/tf_estimator.ckpt')
                            print('Model saved')
                            print('Best Loss ', bestLoss)
                        out_file.flush()

                    if i > maxEpoch * 4 / 5 and val_loss < bestLoss:
                        bestLoss = val_loss
                        saver.save(sess, saveDir + '/tf_estimator.ckpt')
                        print('Model saved')
                        print('Best Loss ', bestLoss)

        out_file.close()
        print('Best Loss ', bestLoss)


    def load_model(self, saveDir, batchSize=1, xshape=[128, 128, 1], yshape=[128, 128, 3]):
        # init model
        # generate model
        self.batchSize = batchSize
        self.xshape = xshape
        self.yshape = yshape
        self.predict_op, self.x = self.build_model(bTrain=False)
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        sess = tf.Session(config=config)
        with tf.device(self.device):
            ckpt = tf.train.get_checkpoint_state(saveDir)
            if ckpt and ckpt.model_checkpoint_path:
                print('loading model ...')
                saver.restore(sess, saveDir + '/tf_denoiser.ckpt')
                self.sess = sess

    def denoise_bayer(self, image, psize, crop):
        """
        denoise a bayer image, whose pixels values are in [0, 1]
        :param image  the image to be denoised
        :param noise: estimated noise level of the image
        :param psize: size of patch
        :param crop:  crop of image patch
        :return:
        """
        assert image.ndim == 3
        start_time = time.time()

        h, w = image.shape[:2]

        psize = min(min(psize, h), w)
        psize -= psize % 2

        # psize = 1024

        patch_step = psize
        patch_step -= 2 * crop
        # patch_step = 4096
        shift_factor = 2

        # Result array
        R = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.float32)

        rangex = range(0, w - 2 * crop, patch_step)
        rangey = range(0, h - 2 * crop, patch_step)
        ntiles = len(rangex) * len(rangey)

        #image = image[..., np.newaxis]

        # resize input
        sess = self.sess
        with tf.device(self.device):
            with tqdm.tqdm(total=ntiles, unit='tiles', unit_scale=True) as pbar:
                for start_x in rangex:
                    for start_y in rangey:
                        a_time = time.time()

                        end_x = start_x + psize
                        end_y = start_y + psize
                        if end_x > w:
                            end_x = w
                            end_x = shift_factor * ((end_x) / shift_factor)
                            start_x = end_x - psize
                        if end_y > h:
                            end_y = h
                            end_y = shift_factor * ((end_y) / shift_factor)
                            start_y = end_y - psize

                        tileM = image[np.newaxis, start_y:end_y, start_x:end_x, :]
                        if DEBUG:
                            print('tileM.shape : ', tileM.shape)

                        b_time = time.time()
                        out = sess.run(self.predict_op, feed_dict={self.x: tileM })
                        c_time = time.time()

                        out = out.reshape(out.shape[1], out.shape[2], 1)
                        s = out.shape[0]
                        R[start_y + crop:start_y + crop + s,
                        start_x + crop:start_x + crop + s, :] = out

                        d_time = time.time()

                        pbar.update(1)

                        if DEBUG:
                            print('image crop : ', (b_time - a_time) * 1000, ' ms')
                            print('forward : ', (c_time - b_time) * 1000, ' ms')
                            print('put patch back :', (d_time - c_time) * 1000, ' ms')

        R[R < 0] = 0.0
        R[R > 1] = 1.0

        runtime = (time.time() - start_time) * 1000  # in ms

        return R, runtime


#######################################################
# Functions to call Estimator

def mem_divide(x, divider):
    # a memory efficient divide function
    # when x is huge, this method saves memory

    for i in range(0, x.shape[0]):
        x[i,...] = x[i, ...] / divider
    return x


def train(modelPath, trainPath, valPath, feature_dim, depth, minNoiseLevel, maxNoiseLevel, x_shape=[128,128,1], y_shape=[128,128,3], device='0'):
    """
    Training using Estimator class.
    :param modelPath: path to save trained model
    :param trainPath: path to training dataset
    :param valPath: path to validation dataset
    :param feature_dim: width of the DNN
    :param depth: depth of the DNN
    :param minNoiseLevel: minimum noise level added to clean images
    :param maxNoiseLevel: maximum noise level added to clean images
    :param x_shape: Input patch size
    :param y_shape: Output patch size
    :param device: which GPU to use (for machines with multiple GPUs, this avoid taking up all GPUs)
    :return: Null
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    estimator = Estimator(device='/gpu:0', depth= depth, feature_dim=feature_dim, xshape=x_shape, yshape=y_shape)

    dc = DatabaseCreator()

    name = 'rgb'
    # res_name = 'gray'
    maxEpoch = 3000

    valY = dc.load_hdf5_v1(valPath, name)
    valY = valY[:estimator.batchSize, ...]

    valY = mem_divide(valY, 255.0)

    npart = dc.load_hdf5_v1(trainPath, 'npart')

    curEpoch = 0
    if os.path.isfile(modelPath + '/loss.txt'):
        with open(modelPath + '/loss.txt', 'r') as log_file:
            log = log_file.readlines()
            if len(log) > 0:
                curEpoch = int(log[-1].split(' ')[0])

    for i in range((curEpoch+1) / (maxEpoch/npart), npart):
    #for i in range(0, 1):
        print('Data part ', i)
        if i > 0:
            final_name = '%s_%d' % (name, i)
            #final_res_name = '%s_%d' % (res_name, i)
        else:
            final_name = name
            #final_res_name = res_name

        trY = dc.load_hdf5_v1(trainPath, final_name)
        trY = mem_divide(trY, 255.0)

        estimator.train(modelPath,  trY,  valY, minNoiseLevel, maxNoiseLevel, maxEpoch=maxEpoch / npart * (i+1))
        #estimator.train(modelPath, trY, valY, minNoiseLevel, maxNoiseLevel, maxEpoch=maxEpoch)

    # estimator.sess.close()

def test(modelPath, feature_dim, depth, device, noise):
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
    inputFolder = 'data/mcm'

    psize = 500
    noise_level = noise / 255.0

    print('true noise : ', noise)
    max_value = 255.0

    crop = 0
    n = 0

    avg_en = 0

    for d, dirs, files in os.walk(inputFolder):
        for f in files:
            if regexp.match(f):
                print('image', n,  f)

                image = cv2.imread(os.path.join(d, f))
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                #cv2.imwrite('%s/%s_rgb.png' % ('output', f), image)

                image = image + np.random.randn(image.shape[0], image.shape[1], image.shape[2]) * noise

                if DEBUG:
                    print ('image.shape : ', image.shape)

                if n == 0:
                    xshape = [psize, psize, 3]
                    yshape = [psize, psize, 3]
                    estimator.load_model(modelPath, batchSize=1, xshape = xshape, yshape=yshape)

                #cv2.imwrite('%s/%s_in.bmp' % ('output', f), np.uint8(image / max_value * 255.0 + 0.5))
                image = image / max_value


                #cv2.imwrite('%s/%s_in.png' % ('output', f), np.uint8(image * 255 + 0.5))

                R, runtime = estimator.denoise_bayer(image, psize, crop)
                out = np.uint8(R * 255 + 0.5)

                estimated_noise = np.mean(np.mean(np.mean(R, axis=0), axis=0), axis=0)

                if DEBUG:
                    print('max value = ', np.max(np.abs(R)))
                    print('time : ', runtime, ' ms')

                #cv2.imwrite('data/dnn_res.bmp', out)

                print('estimate_noise : ', estimated_noise * 255.0)
                cv2.imwrite('%s/%s.png' % ('output', f), out)
                with open('data/time.txt', 'w') as out_file:
                    print(runtime, file=out_file)
                n += 1
                avg_en += estimated_noise

    print('avg_en : ', avg_en / n * 255.0)
    estimator.sess.close()

def test_real(modelPath, feature_dim, depth, device):
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
    inputFolder = 'data/real'

    psize = 500
    max_value = 255.0

    crop = 0
    n = 0

    avg_en = 0

    for d, dirs, files in os.walk(inputFolder):
        for f in files:
            if regexp.match(f):
                print('image', n,  f)

                image = cv2.imread(os.path.join(d, f))
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                #cv2.imwrite('%s/%s_rgb.png' % ('output', f), image)
                if DEBUG:
                    print ('image.shape : ', image.shape)

                if n == 0:
                    xshape = [psize, psize, 3]
                    yshape = [psize, psize, 3]
                    estimator.load_model(modelPath, batchSize=1, xshape = xshape, yshape=yshape)

                #cv2.imwrite('%s/%s_in.bmp' % ('output', f), np.uint8(image / max_value * 255.0 + 0.5))
                image = image / max_value


                # cv2.imwrite('%s/%s_in.png' % ('output', f), np.uint8(image * 255 + 0.5))

                R, runtime = estimator.denoise_bayer(image, psize, crop)
                # out = np.uint8(R * 255 + 0.5)

                estimated_noise = np.mean(np.mean(np.mean(R, axis=0), axis=0), axis=0)

                if DEBUG:
                    print('max value = ', np.max(np.abs(R)))
                    print('time : ', runtime, ' ms')

                #cv2.imwrite('data/ne_res.png', out)
                with h5py.File('data/ne_res.h5', "w") as outFile:
                    outFile.create_dataset('out', data=R * 255, compression='gzip')

                print('estimate_noise : ', estimated_noise * 255.0)
                # cv2.imwrite('%s/%s.png' % ('output', f), out)
                with open('data/time.txt', 'w') as out_file:
                    print(runtime, file=out_file)
                n += 1
                avg_en += estimated_noise

    print('avg_en : ', avg_en / n * 255.0)
    estimator.sess.close()

def test(modelPath, feature_dim, depth, device, noise, use_scalar_noise=True):
    """
    Denoise noisy images using Denoiser class with pre-trained model.
    :param modelPath: path to save trained model
    :param feature_dim: width of the DNN
    :param depth: depth of the DNN
    :param device: which GPU to use (for machines with multiple GPUs, this avoid taking up all GPUs)
    :param noise: standard variation of noise of the tested images
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    denoiser = Denoiser(batchSize=1, feature_dim=feature_dim, depth=depth, use_scalar_noise=use_scalar_noise)
    regexp = re.compile(r'.*\.(%s)' % '(jpg)|(png)')
    inputFolder = 'data'

    psize = 500
    noise_level = noise / 255.0

    print('noise_level: ', noise_level)
    max_value = 255.0

    crop = 0
    n = 0

    dc = DatabaseCreator()

    for d, dirs, files in os.walk(inputFolder):
        for f in files:
            if regexp.match(f):
                print('image', n,  f)

                image = cv2.imread(os.path.join(d, f))
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                cv2.imwrite('%s/%s_rgb.png' % ('output', f), image)

                image = image + np.random.randn(image.shape[0], image.shape[1], image.shape[2]) * noise
                image = dc.rgb2bayer3d(image)

                if DEBUG:
                    print ('image.shape : ', image.shape)

                if n == 0:
                    xshape = [psize, psize, 3]
                    yshape = [psize, psize, 3]
                    denoiser.load_model(modelPath, batchSize=1, xshape = xshape, yshape=yshape)

                #cv2.imwrite('%s/%s_in.bmp' % ('output', f), np.uint8(image / max_value * 255.0 + 0.5))
                image = image / max_value


                cv2.imwrite('%s/%s_in.png' % ('output', f), np.uint8(image * 255 + 0.5))

                R, runtime = denoiser.denoise_bayer(image, noise_level, psize, crop)
                out = np.uint8(R * 255 + 0.5)

                #print('out.shape = ', out.shape)

                if DEBUG:
                    print('max value = ', np.max(np.abs(R)))
                    print('time : ', runtime, ' ms')

                #cv2.imwrite('data/dnn_res.bmp', out)
                cv2.imwrite('%s/%s.png' % ('output', f), out)
                with open('data/time.txt', 'w') as out_file:
                    print(runtime, file=out_file)
                n += 1
    denoiser.sess.close()


if __name__ == '__main__':
    ## configuration
    bTrain = False
    modelPath = 'ne_w64d16_v2_sigma0_30'
    width = 64
    depth = 16 - 4
    device = '0'
    minNoiseLevel = 0.0 / 255.0
    maxNoiseLevel = 30.0 / 255.0
    #### end configuration

    if bTrain:
        train('models/%s' % modelPath, 'data/pristine_rgb2gray.h5',
              'data/kodak_rgb2gray.h5', width, depth, minNoiseLevel, maxNoiseLevel, device=device, x_shape=[128, 128, 3],
              y_shape=[128, 128, 3])
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--noise', type=float, default=0.0,
                            help='standard deviation of additive Gaussian noise, w.r.t to a [0,1] intensity scale.')
        args = parser.parse_args()

        noise = 5

        test('models/%s' % modelPath, width, depth=depth, device=device, noise=noise)
        #test_real('models/%s' % modelPath, width, depth=depth, device=device)








