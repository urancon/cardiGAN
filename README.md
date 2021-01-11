# cardiGAN : Echocardiogram generation using SPADE

INSA Lyon - TDSI - Binôme 10

Ulysse RANÇON, Corentin VANNIER

## Overview

This project aims at generating artificial echocardiogram images using Generative Adversarial Networks (GANs), and more specifically, NVIDIA's [SPADE](https://arxiv.org/abs/1903.07291).
Then, using the generated images to try and improve segmentation results.

More information about the project and how to run it can be found in the Jupyter Notebook included in this repository.

## About the dataset

We are using the CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation) dataset and consists of 
clinical exams from 500 patients, acquired at the University Hospital of St Etienne (France).

The dataset is available [here](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html), and there is currently a [challenge](http://camus.creatis.insa-lyon.fr/challenge/#challenges) running on it. It has the following architecture:

```
.
├── train
│   ├── patient0001
│   │   ├── 2CH
│   │   │   ├── gt
│   │   │   ├── gt_ref
│   │   │   ├── im
│   │   │   └── im_ref
│   │   └── 4CH
│   ├── patient0002
│   └── ...
├── test
│   └── ...
└── valid
    └── ...
```

There is a total 500 patients. For each of those patient, there are two planes of acquisition: 2 chambers (`2CH`) and 4 chambers (`4CH`).

For both of those planes, `im` contains the echocardiography picture, and `gt` its corresponding mask. Those are 256x256 pictures, with two channels corresponding to the end-diastolic and end-systolic views. Their shape is `(2, 256, 256)`. 

*Note: if needed, `im_ref` and `gt_ref` contain the raw data.*

## Further readings

Here are some links that proved useful during the course of our project:
* [an excellent series of articles about GauGAN](https://blog.paperspace.com/nvidia-gaugan-introduction/)
* [this article about checkerboard artifacts coming from deconvolution](https://distill.pub/2016/deconv-checkerboard/)
