% -----------------------------------------------------------------------     
%
% Last revision: 1-Dec-2015
%
% Authors: Guangyong Chen
% License: MIT License
%
% Copyright (c) 2015 Guangyong Chen
%
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this software and associated documentation files (the
% "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to
% permit persons to whom the Software is furnished to do so, subject to
% the following conditions:
% 
% The above copyright notice and this permission notice shall be
% included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
% NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
% LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
% OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
% WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%
% -----------------------------------------------------------------------
clc;clear;close all;
addpath(genpath('./'));
fileList = dir('./imdata/*.jpg');

SigmaCnt = 1;
SigVariance = [];
for givsigma =  10:10:50
    
    for ImCnt = 1: numel(fileList)
        im =  double(imread(fileList(ImCnt).name));
        im_noisy = im + +randn(size(im))*givsigma;

        estsigma =NoiseEstimation(im_noisy, 8);
        
        disp(strcat('gnd:',num2str(givsigma),'-Estimated:',num2str(estsigma)));
    end
end


