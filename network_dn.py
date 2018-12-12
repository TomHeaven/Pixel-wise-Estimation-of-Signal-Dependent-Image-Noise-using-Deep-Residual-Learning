# coding=utf-8
from __future__ import print_function
import torch
import numpy as np
from torch.autograd import  Variable
from torch import nn
import torch.nn.functional as F

debug = False

EPS = 1e-4


class BasicBlock(nn.Module):
    def __init__(self, input_dim, width, block_depth):
        super(BasicBlock, self).__init__()

        self.block_depth = block_depth

        self.conv1 = nn.Conv2d(input_dim, width, kernel_size=3, padding=1)
        if block_depth > 1:
            self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        if block_depth > 2:
            self.conv3 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        if block_depth > 3:
            self.conv4 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        if block_depth > 4:
            raise BaseException('block_depth > 4 is not implemented.')

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out1 = out
        if self.block_depth > 1:
            out = F.relu(self.conv2(out))
        if self.block_depth > 2:
            out = F.relu(self.conv3(out))
        if self.block_depth > 3:
            out = F.relu(self.conv4(out))
        return out + out1




class Network_dn(nn.Module):
    def __init__(self, block_depth=4, block_num=5, width=64, input_dim=3, min_noise_level = 0, max_noise_level = 50,
                 btrain=False):
        super(Network_dn, self).__init__()

        self.block_depth = block_depth
        self.block_num = block_num
        self.input_dim = input_dim
        self.min_noise_level = min_noise_level
        self.max_noie_level = max_noise_level
        self.btrain = btrain
        self.width = width

        self.conv_first = nn.Sequential(
            nn.Conv2d(input_dim, width, kernel_size=3, padding=1),
            nn.ReLU()
        )

        print('block depth : ', block_depth)

        self.conv_block_first = nn.Conv2d(width + 1, width, kernel_size=3, padding=1)
        self.res_block1 = BasicBlock(width+1, width, block_depth)
        self.res_block2 = BasicBlock(width+1, width, block_depth)
        self.res_block3 = BasicBlock(width+1, width, block_depth)
        self.res_block4 = BasicBlock(width+1, width, block_depth)
        self.res_block5 = BasicBlock(width+1, width, block_depth)

        self.conv_block_first = nn.Conv2d(width+1, width, kernel_size=3, padding=1)
        self.conv_last = nn.Sequential(
            nn.Conv2d(width, input_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, noise_level, use_scalar_noise=True):
        # add Gaussian noise
        if self.btrain and noise_level > EPS:
            noise = Variable(torch.randn(x.size()) * torch.from_numpy(np.float32([noise_level])))
            if torch.cuda.is_available():
                noise = noise.cuda()
            x = x + noise

        # construct noise layer
        if use_scalar_noise:
            [B, C, H, W] = x.size()
            noise_layer = torch.ones([B, 1, H, W]) * torch.from_numpy(np.float32([noise_level]))
        else:
            noise_layer = torch.from_numpy(noise_level)
            #print('noise_level.shape : ', noise_level.shape)

        noise_layer = Variable(noise_layer)
        if torch.cuda.is_available():
            noise_layer = noise_layer.cuda()

        # main structure
        h = self.conv_first(x)
        for i in range(self.block_num):
           h = torch.cat([h, noise_layer], dim=1)
           if i == 0:
               h = self.res_block1(h)
           elif i == 1:
               h = self.res_block2(h)
           elif i == 2:
               h = self.res_block3(h)
           elif i == 3:
               h = self.res_block4(h)
           elif i == 4:
               h = self.res_block5(h)

        output = self.conv_last(h)
        return output



def _cvt1(x):
    flag = (x > 0.04045)
    if torch.cuda.is_available():
        flag = flag.type(torch.cuda.FloatTensor)
    else:
        flag = flag.type(torch.FloatTensor)
    return torch.pow((x + 0.055) / 1.055, 2.4) * flag + x / 12.92 * (1 - flag)

def _cvt2(x):
    flag = (x > 0.008856)
    if torch.cuda.is_available():
        flag = flag.type(torch.cuda.FloatTensor)
    else:
        flag = flag.type(torch.FloatTensor)
    return torch.pow(x, 1/3.0) * flag + ((7.787 * x) + 16 / 116) * (1 - flag)

def rgb2lab(rgb):
    """
    convert an image from rgb to lab in pytorch
    :param rgb: input image, values from [0, 1]
    :return:
    """
    r = rgb[:, 0, :, :]
    g = rgb[:, 1, :, :]
    b = rgb[:, 2, :, :]

    # first convert rgb to xyz
    r = _cvt1(r)
    g = _cvt1(g)
    b = _cvt1(b)

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    # then convert xyz to lab
    x = _cvt2(x)
    y = _cvt2(y)
    z = _cvt2(z)

    out = rgb.clone()
    out[:, 0, :, :] = (116 * y) - 16
    out[:, 1, :, :] = 500 * (x - y)
    out[:, 2, :, :] = 200 * (y - z)
    return out

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    image = Variable(torch.randn(5, 3, 128, 128))
    network = Network_dn(n_low=3, n_high=3)
    output = network(image)
    print('output.shape : ', output.shape)






