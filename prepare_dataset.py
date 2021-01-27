# Format the CAMUS dataset for NVlabs/SPADE

# Input: CAMUS .hdf5 file
# Output: three separate folders ('train', 'test', 'valid') each containing two sub-folders ('img', 'label')
# Images are exported to 3-channel .png, and labels to 1-channel .png

import os
import h5py
import numpy as np
from tqdm import tqdm
import cv2

src_path = os.getcwd()
camus_path = "camus05_256.hdf5"

with h5py.File(camus_path, 'r') as f:
    for subset in ['train', 'test', 'valid']:
        i = 0
        print('Processing the ' + subset + ' set...')

        # Directories
        if not os.path.exists(src_path + '/' + subset):
            os.makedirs(src_path + '/' + subset)
            os.chdir(src_path + '/' + subset)
            os.mkdir(src_path + '/' + subset + '/img')
            os.mkdir(src_path + '/' + subset + '/label')
            os.chdir('..')
        img_path = src_path + '/' + subset + '/img/'
        label_path = src_path + '/' + subset + '/label/'

        # Processing
        for patient in tqdm(f[subset]):
            for view in f[subset][patient]:
                for time in [0, 1]:
                    image = f[subset][patient][view]['im'][time, :, :, :]  # image
                    mask = f[subset][patient][view]['gt'][time, :, :]  # mask
                    image = 255 * np.repeat(image.reshape(256, 256, 1), 3, axis=2)  # reshape image to 3-channel RGB

                    file_end = '_ED_img.png' if time == 0 else '_ES_img.png'
                    cv2.imwrite(img_path + patient + '_' + view + file_end, image)
                    cv2.imwrite(label_path + patient + '_' + view + file_end, mask)
                    i += 1

        # Finishing
        print('Processed ' + str(i) + ' images from the ' + subset + ' set.')
        print('\n')
