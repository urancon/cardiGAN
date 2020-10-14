# script to prepare a subset of CAMUS dataset to train our GAN with NVlab's GauGAN/SPADE implementation
# subset : (4 chambers - end of diastole)

# Input: original CAMUS dataset hdf5 file
# Output: three separate folders ('train', 'test', 'valid') each containing ('img', 'label') subfolders
# filled with .png images. Actual images are 3-channel RGB, while annotation files / masks have only 1 channel

# this dataset format is required by NVlab's GauGAN/SPADE repo to train a network on.


import os
import h5py
import numpy as np
from tqdm import tqdm
import cv2


# get current location and load the original dataset hdf5 file
print('\n')
src_path = os.getcwd()
camus_path = "/home/ulysse/Desktop/ProjetGAN/Data/camus05_256.hdf5"
f = h5py.File(camus_path, 'r')


# fill each set with its images and labels
for set in ['train', 'test', 'valid']:

    if not os.path.exists(src_path + '/' + set):
        os.makedirs(src_path + '/' + set)
        os.chdir(src_path + '/' + set)
        os.mkdir(src_path + '/' + set + '/img')
        os.mkdir(src_path + '/' + set + '/label')
        os.chdir('..')

    img_path = src_path + '/' + set + '/img/'
    label_path = src_path + '/' + set + '/label/'

    print('starting gathering ' + set + ' set')
    i = 0  # file counter

    for patient in tqdm(f[set]):

        image = f[set][patient]['4CH']['im'][0, :, :, :]  # patient image - 4 chambers - end of diastole (channel0 only)
        mask = f[set][patient]['4CH']['gt'][0, :, :]  # corresponding mask

        img = np.repeat(image.reshape(256, 256, 1), 3, axis=2)  # must convert 1 channel to 3-channel RGB array
        img = 255 * img
        cv2.imwrite(img_path + patient + '_4CH_ED_img.png', img)

        cv2.imwrite(label_path + patient + '_4CH_ED_img.png', mask)

        i += 1

    print('finished gathering ' + set + ' set')
    print(str(i) + ' images written in ' + set)
    print('\n')

f.close()
