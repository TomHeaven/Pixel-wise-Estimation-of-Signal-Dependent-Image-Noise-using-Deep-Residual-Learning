# Pixel-wise Estimation of Signal Dependent Image Noise using Deep Residual Learning


Hanlin Tan, Huaxin Xiao, Shiming Lai, Yu Liu and Maojun Zhang

National University of Defense Technology, China


## Applying DRNE to Deep Denoising Model

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

## Train Your Own Model
+ Download [Waterloo](https://ece.uwaterloo.ca/~k29ma/exploration) dataset, [Kodak](http://www.cs.albany.edu/~xypan/research/snr/Kodak.html) dataset and [McM](https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm) dataset and put the images in the data folder such as:

```
├── data
│   ├── kodak
│   ├── mcm
│   └── pristine_images
```

+ Run `data_v3.py` to generate cache files of images in `data` folder. If you need to use your own dataset, update function `create_pristine`, accordingly.
+ Update configuration part of `model_noise_estimation_w64d16_v2_sigma0_30.py` by setting `bTrain = True`. For example,

```python
if __name__ == '__main__':
    ## configuration
    bTrain = True
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
```
+ Start training by
```
python model_noise_estimation_w64d16_v2_sigma0_30.py
```

# Reference
If you use the code in your work, please cite our paper:

```
@article{tan2019pixelwise,
  title={Pixelwise Estimation of Signal-Dependent Image Noise Using Deep Residual Learning},
  author={Tan, Hanlin and Xiao, Huaxin and Lai, Shiming and Liu, Yu and Zhang, Maojun},
  journal={Computational intelligence and neuroscience},
  volume={2019},
  year={2019},
  publisher={Hindawi}
}
```


