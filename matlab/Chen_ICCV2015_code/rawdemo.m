
clc;clear;close all;
addpath(genpath('./'));
fileList = dir('./rawdata/*.png');

SigmaCnt = 1;
SigVariance = [];

    
    for ImCnt = 1: numel(fileList)
        im_noisy =  double(imread(fileList(ImCnt).name));
        estsigma =NoiseEstimation(im_noisy, 8);
        fprintf('File: %s Sigma: %.2f\n', fileList(ImCnt).name, estsigma);
    end



