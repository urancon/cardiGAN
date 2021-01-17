# cardiGAN: Towards better echocardiogram segmentation by data generation using SPADE

INSA Lyon - TDSI - Binôme 10

Ulysse RANÇON, Corentin VANNIER


## Overview

This project aims at generating artificial echocardiogram images using Generative Adversarial Networks (GANs), and more specifically, NVIDIA's [SPADE](https://arxiv.org/abs/1903.07291).
We then used the generated images with their annotation as additional training data to try and improve the performances of semantic segmentation models.

More information about the project and how to run it can be found in the [Jupyter Notebook](https://github.com/urancon/cardiGAN/blob/master/cardiGAN.ipynb) included in this repository.

You can also see [our report](https://github.com/urancon/cardiGAN/blob/master/cardiGAN.pdf) for deeper insights of our work.


## About the dataset

We are using the **CAMUS** (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation) dataset.
It consists of clinical exams from 500 patients, acquired at the University Hospital of St Etienne (France).

The dataset is available [here](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html), and a [challenge](http://camus.creatis.insa-lyon.fr/challenge/#challenges) is currently running on it. It has the following architecture:

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
│   │       └── ...
│   ├── patient0002
│   └── ...
├── test
│   └── ...
└── valid
    └── ...
```

There is a total 500 patients. For each of those patient, there are two planes of acquisition: 2 chambers (`2CH`) and 4 chambers (`4CH`).

For both of those planes, `im` contains the echocardiography picture, and `gt` its corresponding mask. Those are 256x256 pictures, with two channels corresponding to the end-diastolic (ED) and end-systolic (ES) views. Their shape is `(2, 256, 256)`. 

*Note: if needed, `im_ref` and `gt_ref` contain the raw data.*


## Citation

If you happen to like our work, here is a citation record you can use:
```latex
@ARTICLE{cardiGAN,
       author = {{Rancon} Ulysse and {Vannier} Corentin and {Bernard} Olivier},
        title = "{CardiGAN: towards better echocardiogram segmentation by data generation using SPADE}",
         year = 2021,
        month = jan
}
```
