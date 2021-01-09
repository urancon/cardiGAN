# script to prepare a subset of CAMUS dataset to train our GAN with NVLab's GauGAN/SPADE implementation
# subset : (4 chambers - end of diastole + end of systole)

# Input: original CAMUS dataset hdf5 file
# Output: three separate folders ('train', 'test', 'valid') each containing ('img', 'label') sub-folders
# filled with .png images. Actual images are 3-channel RGB, while annotation files / raw mhd have only 1 channel

# this dataset format is required by NVLab's GauGAN/SPADE repo to train a network on.


import os
import h5py
import numpy as np
from tqdm import tqdm
import cv2


# get current location and load the original dataset hdf5 file
print('\n')
src_path = os.getcwd()
camus_path = "camus05_256.hdf5"
f = h5py.File(camus_path, 'r')


# fill each set with its images and labels
for subset in ['train', 'test', 'valid']:

    if not os.path.exists(src_path + '/' + subset):
        os.makedirs(src_path + '/' + subset)
        os.chdir(src_path + '/' + subset)
        os.mkdir(src_path + '/' + subset + '/img')
        os.mkdir(src_path + '/' + subset + '/label')
        os.chdir('..')

    img_path = src_path + '/' + subset + '/img/'
    label_path = src_path + '/' + subset + '/label/'

    print('starting gathering ' + subset + ' set')
    i = 0  # file counter

    for patient in tqdm(f[subset]):

        fch_ed_image = f[subset][patient]['4CH']['im'][0, :, :, :]  # 4 chambers - end of diastole (channel 0)
        fch_ed_mask = f[subset][patient]['4CH']['gt'][0, :, :]  # corresponding mask
        fch_ed_image = 255 * np.repeat(fch_ed_image.reshape(256, 256, 1), 3, axis=2)  # must convert 1 channel to 3-channel RGB array
        cv2.imwrite(img_path + patient + '_4CH_ED_img.png', fch_ed_image)
        cv2.imwrite(label_path + patient + '_4CH_ED_img.png', fch_ed_mask)

        fch_es_image = f[subset][patient]['4CH']['im'][1, :, :, :]  # 4 chambers - end of systole (channel 1)
        fch_es_mask = f[subset][patient]['4CH']['gt'][1, :, :]  # corresponding mask
        fch_es_image = 255 * np.repeat(fch_es_image.reshape(256, 256, 1), 3, axis=2)  # must convert 1 channel to 3-channel RGB array
        cv2.imwrite(img_path + patient + '_4CH_ES_img.png', fch_es_image)
        cv2.imwrite(label_path + patient + '_4CH_ES_img.png', fch_es_mask)

        tch_ed_image = f[subset][patient]['2CH']['im'][0, :, :, :]  # 2 chambers - end of diastole (channel 0)
        tch_ed_mask = f[subset][patient]['2CH']['gt'][0, :, :]  # corresponding mask
        tch_ed_image = 255 * np.repeat(tch_ed_image.reshape(256, 256, 1), 3, axis=2)  # must convert 1 channel to 3-channel RGB array
        cv2.imwrite(img_path + patient + '_2CH_ED_img.png', tch_ed_image)
        cv2.imwrite(label_path + patient + '_2CH_ED_img.png', tch_ed_mask)

        tch_es_image = f[subset][patient]['2CH']['im'][1, :, :, :]  # 2 chambers - end of systole (channel 1)
        tch_es_mask = f[subset][patient]['2CH']['gt'][1, :, :]  # corresponding mask
        tch_es_image = 255 * np.repeat(tch_es_image.reshape(256, 256, 1), 3, axis=2)  # must convert 1 channel to 3-channel RGB array
        cv2.imwrite(img_path + patient + '_2CH_ES_img.png', tch_es_image)
        cv2.imwrite(label_path + patient + '_2CH_ES_img.png', tch_es_mask)

        i += 4

    print('finished gathering ' + subset + ' set')
    print(str(i) + ' images written in ' + subset)
    print('\n')

f.close()
