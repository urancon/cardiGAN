# Format the CAMUS dataset for NVlabs/SPADE

# Input: CAMUS .hdf5 file
# Output: three separate folders ('train', 'test', 'valid') each containing two sub-folders ('img', 'label')
# Images are exported to 3-channel .png, and labels to 1-channel .png

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
        for view in f[subset][patient]:
            for instant in [0, 1]:
                image = f[subset][patient][view]['im'][instant, :, :, :]  # image
                mask = f[subset][patient][view]['gt'][instant, :, :]  # corresponding mask
                image = 255 * np.repeat(image.reshape(256, 256, 1), 3, axis=2)  # must convert 1 channel to 3-channel RGB array

                file_end = '_ED_img.png' if instant == 0 else '_ES_img.png'
                cv2.imwrite(img_path + patient + '_' + view + file_end, image)
                cv2.imwrite(label_path + patient + '_' + view + file_end, mask)
                i += 1

    print('finished gathering ' + subset + ' set')
    print(str(i) + ' images written in ' + subset)
    print('\n')

f.close()
