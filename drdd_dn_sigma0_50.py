# coding=utf-8
from __future__ import  print_function
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import os
from data_v3 import DatabaseCreator
import numpy as np
import time
import tqdm
import argparse
import tifffile as tiff
import re
import cv2

from network_dn import Network_dn


# config
DEBUG = False

def print_net(net):
    for i, weights in enumerate(list(net.parameters())):
        print('i:', i, 'weights:', weights.size())


class DeepProcesser:
    """
    A class to train and test a tensorflow denoiser.
    """

    # predict_op = []
    def __init__(self, block_depth=4, block_num=5, width = 64, min_noise_level=0, max_noise_level=30,
                 device = '/gpu:0', batch_size = 32, input_dim = 3, lr=1e-4, use_scalar_noise=True):
        self.batchSize = batch_size
        self.block_num = block_num
        self.block_depth = block_depth
        self.width = width
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.device = device
        self.input_dim = input_dim
        self.lr = lr
        self.use_scalar_noise=use_scalar_noise

    def train(self, model_dir, trY,  valY, maxEpoch=1000, part=0):
        """
        train
        :param trX:
        :param trY:
        :param maxEpoch:
        :param batchSize:
        :return:
        """

        # add new axis for gray image data
        if trY.ndim == 3:
            trY = trY[..., np.newaxis]
        if valY.ndim == 3:
            valY = valY[..., np.newaxis]

        # generate model
        if not hasattr(self, 'net'):
            print('Building training model ...')
            net = Network_dn(block_depth=self.block_depth, block_num=self.block_num, width=self.width,
                             input_dim=self.input_dim, btrain=True)
            if torch.cuda.is_available():
                net.cuda()
            print_net(net)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
            loss_func = F.mse_loss
            #loss_func = IspLoss_v2(alpha=0.5)

        # prepare model folder and log file
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        curEpoch = 0
        bestLoss = 99999.0
        model_path = model_dir + '/deep_isp.pickle'
        log_path = model_dir + '/loss.txt'


        # restore trainning according to log
        if os.path.isfile(log_path):
            with open(log_path, 'r') as log_file:
                log = log_file.readlines()
                if len(log) > 0:
                    curEpoch = int(log[-1].split(' ')[0]) + 1 + part * maxEpoch
        out_file = open(log_path, 'a')


        if os.path.isfile(model_path):
                print('Restored training...')
                net.load_state_dict(torch.load(model_path))
        else:
                print('Start training...')

        for i in range(curEpoch, maxEpoch):
            start_time = time.time()
            print('Epoch %d ...' % i)

            train_loss = 0
            cnt = 0
            # print('len trY : ', len(trY), ' len trX : ', len(trX))
            for start, end in zip(range(0, len(trY), self.batchSize),
                                  range(self.batchSize, len(trY) + 1, self.batchSize)):

                n_level = np.random.rand(1) * (self.max_noise_level - self.min_noise_level) + self.min_noise_level
                y = trY[start:end]
                y = Variable(torch.from_numpy(y))

                if torch.cuda.is_available():
                    y = y.cuda()

                output = net(y, n_level)
                loss = loss_func(output, y)
                train_loss += loss
                # update weights
                net.zero_grad()
                loss.backward()
                optimizer.step()
                cnt += 1

            train_loss /= cnt
            if torch.cuda.is_available():
                train_loss = train_loss.cpu()
            train_loss = train_loss.data.numpy()[0]
            print('train_loss : ', train_loss)

            for n_level in [5]:
                val_loss = 0
                cnt = 0
                for start, end in zip(range(0, len(valY), self.batchSize),
                                      range(self.batchSize, len(valY) + 1, self.batchSize)):
                    y = valY[start:end]
                    y = Variable(torch.from_numpy(y), volatile=True)

                    if torch.cuda.is_available():
                        y = y.cuda()

                    output = net(y, n_level / 255.0)
                    val_loss += loss_func(output, y)
                    cnt += 1
                    # prevent too much computation that wastes training time
                    if cnt > 10:
                        break

                # print loss
                val_loss /= cnt

                if torch.cuda.is_available():
                    val_loss = val_loss.cpu()

                val_loss = val_loss.data.numpy()[0]

                print('noise_level : ', n_level, ' val_loss : ', val_loss)
                print(i, n_level, train_loss, val_loss, file=out_file)

            print('time : ', time.time() - start_time, ' s')



            if i % 10 == 0 and i < maxEpoch * 4 / 5:
                torch.save(net.state_dict(), model_path)
                print('Model saved')
                if val_loss < bestLoss:
                    bestLoss = val_loss
                    print('Best Loss ', bestLoss)
                out_file.flush()

            if i > maxEpoch * 4 / 5 and val_loss < bestLoss:
                bestLoss = val_loss
                torch.save(net.state_dict(), model_path)
                print('Model saved')
                print('Best Loss ', bestLoss)

        out_file.close()
        print('Best Val Loss ', bestLoss)


    def load_model(self, model_dir, btrain=False):
        print('Building model ...')
        net = Network_dn(self.block_depth, self.block_num, self.width, self.input_dim, btrain=btrain)
        #print('block_num : ', self.block_num, ' block_depth : ', self.block_depth)
        print('Loading model ...')
        net.load_state_dict(torch.load('%s/deep_isp.pickle' % model_dir))
        if torch.cuda.is_available():
            net.cuda()
        self.net = net
        print_net(net)

    def test(self, image, noise_level, psize, crop):
        """
        denoise a bayer image, whose pixels values are in [0, 1]
        :param image  the image to be denoised
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

        image = np.float32(image)

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
                    tileM = tileM.transpose(0, 3, 1, 2)
                    tileM = Variable(torch.from_numpy(tileM))
                    if torch.cuda.is_available():
                        tileM = tileM.cuda()

                    if DEBUG:
                        print('tileM.shape : ', tileM.shape)

                    b_time = time.time()

                    if self.use_scalar_noise:
                        out = self.net(tileM, noise_level, self.use_scalar_noise)
                    #if True:
                    #    out = self.net(tileM, 5.0 / 255.0, True)
                    else:
                        noise = noise_level[:, :, start_y:end_y, start_x:end_x]
                        #noise = np.ones((noise.shape[0], noise.shape[1], noise.shape[2], noise.shape[3]), dtype=np.float32) * 5.0 / 255.0
                        out = self.net(tileM, noise, self.use_scalar_noise)


                    c_time = time.time()

                    if torch.cuda.is_available():
                        out = out.data.cpu()
                    else:
                        out = out.data

                    out = out.numpy().transpose(0, 2, 3, 1)
                    out = out[0, ...]

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



def mem_divide(x, divider):
    # a memory efficient divide function
    # when x is huge, this method saves memory

    for i in range(0, x.shape[0]):
        x[i,...] = x[i, ...] / divider
    return x


def train(modelPath, trainPathY, valPathY, block_num, block_depth, width, minNoiseLevel, maxNoiseLevel, device='0'):
    """
    Training using Denoiser class.
    :param modelPath: path to save trained model
    :param trainPath: path to training dataset
    :param valPath: path to validation dataset
    :param feature_dim: width of the DNN
    :param depth: depth of the DNN
    :param minNoiseLevel: minimum noise level added to clean images
    :param maxNoiseLevel: maximum noise level added to clean images
    :param device: which GPU to use (for machines with multiple GPUs, this avoid taking up all GPUs)
    :return: Null
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    deepProcesser = DeepProcesser(block_depth=block_depth, block_num=block_num, width=width,
                                  min_noise_level=minNoiseLevel, max_noise_level=maxNoiseLevel, device='/gpu:0')

    dc = DatabaseCreator()
    nameY = 'rgb'
    # res_name = 'gray'
    maxEpoch = 1000

    npart = dc.load_hdf5_v1(trainPathY, 'npart')
    print('npart : ', npart)
    # loading validation data

    final_nameY = '%s_%d' % (nameY, npart - 1)

    valY = dc.load_hdf5_v1(trainPathY, final_nameY)

    valY = valY[:deepProcesser.batchSize, ...]

    valY = mem_divide(valY, 255.0)

    # change dims and float64 to float32 for pytorch
    valY = np.float32(valY.transpose(0, 3, 1, 2))

    curEpoch = 0
    if os.path.isfile(modelPath + '/loss.txt'):
        with open(modelPath + '/loss.txt', 'r') as log_file:
            log = log_file.readlines()
            if len(log) > 0:
                curEpoch = int(log[-1].split(' ')[0])
    # loading training data
    for i in range((curEpoch + 1) / (maxEpoch / (npart - 1)), npart - 1):
        print('Data part ', i)
        if i > 0:
            final_nameY = '%s_%d' % (nameY, i)
        else:
            final_nameY = nameY

        trY = dc.load_hdf5_v1(trainPathY, final_nameY)
        trY = mem_divide(trY, 255.0)

        # change dims and float64 to float32 for pytorch
        trY = np.float32(trY.transpose(0, 3, 1, 2))

        deepProcesser.train(modelPath, trY, valY, maxEpoch=maxEpoch / (npart - 1) * (i + 1))

def test(model_dir, block_num, width, block_depth, device, noise, outfile, use_scalar_noise=True):
    """
    Denoise noisy images using Denoiser class with pre-trained model.
    :param modelPath: path to save trained model
    :param device: which GPU to use (for machines with multiple GPUs, this avoid taking up all GPUs)
    :param noise: standard variation of noise of the tested images
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    deepProcesser = DeepProcesser(block_num=block_num, width=width, block_depth=block_depth,use_scalar_noise=use_scalar_noise)
    deepProcesser.load_model(model_dir, False)

    regexp = re.compile(r'.*\.(%s)' % '(tif|tiff|jpg|png)')
    inputFolder = 'data/medical'

    psize = 200
    noise_level = noise / 255.0
    max_value = 255.0

    crop = 0
    n = 0

    #dc = DatabaseCreator(inBayerType='grbg', outBayerType='grbg')

    for d, dirs, files in os.walk(inputFolder):
        for f in files:
            if regexp.match(f):
                print('image', n,  f)
                image = cv2.imread(os.path.join(d, f))

                if DEBUG:
                    print ('image.shape : ', image.shape)

                image = image / max_value
                R, runtime = deepProcesser.test(image, noise_level, psize, crop)
                out = np.uint8(R * 255 + 0.5)

                #R = swap_blue_red(R)

                if DEBUG:
                    print('max value = ', np.max(np.abs(R)))
                    print('time : ', runtime, ' ms')
                cv2.imwrite(outfile, out)
                with open('data/time.txt', 'w') as out_file:
                    print(runtime, file=out_file)
                n += 1

if __name__ == '__main__':
    ## configuration
    bTrain = True
    modelPath = 'dn_sigma0_50'
    block_num = 5
    block_depth = 4
    width = 64
    device = '0'
    minNoiseLevel = 0.0 / 255.0
    maxNoiseLevel = 50.0 / 255.0
    #### end configuration

    if bTrain:
        train('models/%s' % modelPath, '/Volumes/文档/实验数据/Denoiser/pristine_rgb2gray.h5',
              '/Volumes/文档/实验数据/Denoiser/kodak_rgb2gray.h5', block_num,  block_depth, width, minNoiseLevel,
              maxNoiseLevel, device=device)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--noise', type=float, default=0.0,
                            help='standard deviation of additive Gaussian noise, w.r.t to a [0,1] intensity scale.')

        parser.add_argument('--model_name', type=str, default=None, help='path to model')
        parser.add_argument('--outfile', type=str, default=None, help='path to model')
        parser.add_argument('--gpu', type=int, default=1, help='use GPU or not')

        args = parser.parse_args()

        if not args.gpu == 1:
            device = ''

        print('model_name : ', args.model_name)
        print('use_gpu : ', args.gpu)

        args.outfile = 'test_res.png'
        args.noise = 8

        #test('DRDD_pytorch/models/%s' % modelPath, block_num, width, block_depth, device=device, noise=args.noise,
        #     outfile=args.outfile)
        test('models/%s' % modelPath, block_num, width, block_depth, device=device, noise=args.noise,
             outfile=args.outfile)




