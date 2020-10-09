# cardiGAN : Echocardiogram generation using SPADE algorithm

Corentin Vannier & Ulysse Rancon

Oct. 2020

## Overview

This project aims at generating artificial echocardiogram images, using Generative Adversarial Networks.
It is based on [NVIDIA SPADE algorithm](https://github.com/NVlabs/SPADE) and runs on PyTorch.

Generated images will then serve as an image segmentation dataset.

## Dataset

The dataset used is called **CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation)** and consists of 
clinical exams from 500 patients, acquired at the University Hospital of St Etienne (France).

It is available [here](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html) and a challenge is running on it, 
as of October 2020.

The original dataset has the following architecture:

```shell script
dataset_camus (hdf5 file)
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
  |    |    |    |___ gt
  |    |    |    |___ gt_ref
  |    |    |    |___ im
  |    |    |    |___ im_ref
  |    | 
  |    |___ ...
  |
  |___ test
  |
  |___ valid
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
train_4CH_FD (hdf5 file)
  |
  |___ patient0001_4CH_im_channel0   
  |           
  |___ patient0002_4CH_im_channel0 
  |           
  |___ patient0003_4CH_im_channel0 
  |
    ...
```

An annotation hdf5 file is generated the same way for the corresponding masks, by using **gt** instead of **im**. 
It has the same architecture.

```shell script
anno_train_4CH_FD (hdf5 file)
  |
  |___ patient0001_4CH_gt_channel0   
  |           
  |___ patient0002_4CH_gt_channel0 
  |           
  |___ patient0003_4CH_gt_channel0 
  |
    ...
```

This is done for train, test, and validation sets, which comprise respectively 400, 50, and 50 images. And we're all set
to train our network !


## Installation instruction



## Further readings
