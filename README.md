# cardiGAN : Echocardiogram generation using SPADE algorithm

Corentin Vannier & Ulysse Rancon

Oct. 2020

## Overview

This project aims at generating artificial echocardiogram images, using Generative Adversarial Networks.
It is based on [NVIDIA SPADE algorithm](https://arxiv.org/abs/1903.07291) and runs on PyTorch.

Generated images will then serve as an image segmentation dataset.

## Dataset

The dataset used is called **CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation)** and consists of 
clinical exams from 500 patients, acquired at the University Hospital of St Etienne (France).

It is available [here](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html) and a challenge is running on it, 
as of October 2020.

The original dataset has the following architecture:

```shell script
.hdf5 file
 |
 |___ train
 |    |
 |    |___ patient0001   
 |    |    |
 |    |    |___ 2CH
 |    |    |    |___ gt
 |    |    |    |___ gt_ref
 |    |    |    |___ im
 |    |    |    |___ im_ref
 |    |    |
 |    |    |___ 4CH
 |    |         |___ gt
 |    |         |___ gt_ref
 |    |         |___ im
 |    |         |___ im_ref
 |    | 
 |    |___ ...
 |
 |___ test
 |    |
 |    |___ ...
 |
 |___ valid
      |
      |___ ...

```

There are 500 patients, each of them gathering 2 planes of acquisition, showing either 2 or 4 chambers (denoted by '2CH' and '4CH').
Within each of these subfolders, **im** represents the actual image captured, and **gt** its corresponding mask. Both have 
shape of (2, 256, 256), the first channel being for the end of diastolic instant, and the second for the end of the systolic 
instant.
**im_ref** and **gt_ref** are just raw data before it was resized to 256x256; we don't use them in our case.

In a first step, we will try and train a GAN on to produce "false" echocardiograms of 4-chambers view. For this purpose, 
we had to modify the dataset to keep only the relevant information. That is, _channel n°0 of 4CH/im and 4Ch/gt for all patients_.
The script ```prepare_dataset.py``` outputs such a subset of the original CAMUS dataset:

```shell script
CAMUS_256_4CH
|
|___ train
|       |___ img
|       |___ label
|
|___ test
|       |___ img
|       |___ label
|
|___ valid
        |___ img
        |___ label  
```

Annotation masks, or labels, are retrieved the same way by using **gt** instead of **im**. 

This is done for train, test, and validation sets, which comprise respectively 400, 50, and 50 images. All three folders are
now compatible with NVlab's GauGAN training framework.


## Installation instruction

First download NVlab's official implementation of GauGAN/SPADE on [their Github repository](https://github.com/NVlabs/SPADE):
```shell script
git clone https://github.com/NVlabs/SPADE
```
Also install the Synchronized-BatchNorm-PyTorch rep as follows:
````shell script
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
````

Create a directory ```CAMUS``` under ```SPADE/datasets/```, download CAMUS dataset (file **camus05_256.hdf5**) inside.

Still under this directory, download this repo's **prepare_dataset_v2.py** and execute it by ```python3 prepare_dataset_v2.py```.
This should prepare the dataset and split it in three folders as explained in the preceding section. Everything is now ready for training.


## Run a training

All training commands should be executed at the root of SPADE directory.
An example command to run a training could be:
```shell script
python3 train.py --name CAMUS_4CH_ED --gpu_ids 0,1 --dataset_mode custom --no_instance --label_nc 3 --contain_dontcare_label \
        --preprocess_mode none --label_dir datasets/CAMUS/train/label --image_dir datasets/CAMUS/train/img \
        --load_size 256 --aspect_ratio 1 --crop_size 256 --ngf 48 --ndf 48 --batchSize 2 \
        --niter 20 --niter_decay 10 --tf_log --beta1 0 --beta2 0.999
```

To get more help on how to set training parameters, run ```python3 train.py --help``` in your terminal.

This is actually the set of hyperparameters we have tried for the moment, inspired by the official SPADE paper. The next step
is now to find the best set for the best performances.


## Testing the model

As for training commands, testing commands should be executed at the root of the SPADE repository. For instance:
```shell script
python3 test.py --name CAMUS_4CH_ED_v2 --dataset_mode custom --no_instance --label_nc 3 --contain_dontcare_label \
        --preprocess_mode none --label_dir datasets/CAMUS/train/label --image_dir datasets/CAMUS/train/img \
        --load_size 256 --aspect_ratio 1 --crop_size 256 --ngf 48 \
        --how_many 15
```
This command runs inference on 15 images of the test set, and saves synthesized images in the ```results/``` directory of 
SPADE's repo. From there, you can compare these fake images with the masks that served to produce them.



## Further readings

Here are some links that proved useful for the development of our GAN. They are especially recommended for beginners 
with GANs like us:
* an [excellent series of articles about GauGAN](https://blog.paperspace.com/nvidia-gaugan-introduction/) (with a tutorial) 
* about [checkerboard artifacts coming from deconvolution](https://distill.pub/2016/deconv-checkerboard/)
* ...
