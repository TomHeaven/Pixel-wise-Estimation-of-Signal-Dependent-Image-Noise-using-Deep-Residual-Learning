# Pixel-wise Estimation of Signal Dependent Image Noise using Deep Residual Learning


Hanlin Tan, Huaxin Xiao, Shiming Lai, Yu Liu and Maojun Zhang

National University of Defense Technology, China

---
This project is temporarily for paper review. The full code and guide will be uploaed once the review process ends. 

---


## Applying DRNE to deep denoising model

The code is written in Python 2.7 using Tensorflow 1.4. To evaluate Chen's method, you also need to install Matlab.

We only include `Kodak` dataset images in `data` folder to reduce the attachment size. You can download `McM` and `BSD500` from the Internet if you want to see results on those two datasets. 



The following steps reproduce the results in Section IV-E.

+ run `generate_noisy_images.py` to genereate noisy dataset from groundtruth images.

+ run `matlab/chenxxx/generate_all.m` to genereate noise estimation results, which will be stored in `data/chen`.

+ run `test_drne_and_dn.py` to generate denoised images with the proposed DRNE + a CNN denoiser.

+ run `test_chen_and_dn.py` to enerate denoised images with the Chen's method + the CNN denoiser.

+ run `evaluate.py` to get the quantitative evaluation results. 

Since we uploaded pre-computed results, you can directly run the last step to get the quantitative evaluation results.

Note we use random numbers in noisy image generation. The absolute performance is expected to show a small difference but the performance gain should stay the same.



