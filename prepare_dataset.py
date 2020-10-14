# script to prepare a subset of CAMUS dataset to train our GAN
# subset : (4 chambres - fin diastole)

# Input: original CAMUS dataset hdf5 file
# Output: three separate hdf5 files ('train', 'test', 'valid') with 1-channel images

import time
import os
import sys
import h5py
import numpy as np
from tqdm import tqdm

print('\n')
camus_path = "/home/ulysse/Desktop/ProjetGAN/Data/camus05_256.hdf5"
f = h5py.File(camus_path, 'r')

for set in ['train', 'test', 'valid']:

    print('starting gathering ' + set + ' set')
    new_dset = set + '_4CH_FD.hdf5'
    i = 0  # file counter

    with h5py.File(new_dset, 'w') as newdatafile:
        with h5py.File("anno_" + new_dset, 'w') as newannofile:
            for patient in tqdm(f[set]):
                image = f[set][patient]['4CH']['im'][0, :, :,
                        :]  # image du patient - 4 chambres - fin diastole (canal 0 seulement)
                mask = f[set][patient]['4CH']['gt'][0, :, :]  # masque correspondant
                newdatafile.create_dataset(patient, data=image)
                newannofile.create_dataset(patient, data=mask)
                i += 1

            print('finished gathering ' + set + ' set')
            print(str(i) + ' images written in ' + new_dset)
            print('\n')

            newannofile.close()
        newdatafile.close()

f.close()
