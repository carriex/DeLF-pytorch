
#-*- coding: utf-8 -*-

'''
dataloader.py
'''

import sys, os, time

import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = 1000000000     # to avoid error "https://github.com/zimeon/iiif/issues/11"
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # to avoid error "https://github.com/python-pillow/Pillow/issues/1510"

class GoogleLandmark(torch.utils.data.Dataset):
    '''google landmark dataset'''
    def __init__(self, data_dir, data_csv, transform, idx_to_class=None):
        self.data_dir = data_dir
        self.data_csv = pd.read_csv(data_csv)
        if idx_to_class is not None:
            self.idx_to_class = idx_to_class
        else:
            self.idx_to_class = self.data_csv.landmark_id.unique()
        self.class_to_idx = {class_name: idx for (idx, class_name) in enumerate(self.idx_to_class)}
        self.num_classes = len(self.idx_to_class)
        self.transform = transform

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        '''return (img_feature, label)'''
        img_id = self.data_csv.loc[idx].id
        label = self.data_csv.loc[idx].landmark_id
        img_path = os.path.join(self.data_dir, str(label), str(img_id)+'.jpg')
        img = Image.open(img_path)
        return self.transform(img), torch.tensor(self.class_to_idx[label])


def get_loader(
    train_path,
    val_path,
    stage,
    train_batch_size,
    val_batch_size,
    sample_size,
    crop_size,
    workers):

    prepro = []
<<<<<<< HEAD
=======
    print(sample_size)
>>>>>>> add retrieval code
    prepro.append(transforms.Resize(size=(sample_size,sample_size)))
    # prepro.append(transforms.CenterCrop(size=sample_size))
    # ------------random crop----------------#
    # prepro.append(transforms.RandomCrop(size=crop_size, padding=0))
    # ------------distort aspect ratio----------------#
    # prepro.append(transforms.RandomPerspective())
    prepro.append(transforms.ToTensor())
    train_transform = transforms.Compose(prepro)
    train_path = train_path

    # for val
    prepro = []
    prepro.append(transforms.Resize(size=(sample_size,sample_size)))
    # prepro.append(transforms.CenterCrop(size=crop_size))
    prepro.append(transforms.ToTensor())
    val_transform = transforms.Compose(prepro)
    val_path = val_path


    
    # image folder dataset.

    #------------hardcode dataset path for now----------------#
    train_dataset = GoogleLandmark(data_dir='/data/google-landmark/org/train',
                                   data_csv='/data/google-landmark/csv/train-clean-1000.csv',
                                   transform=train_transform)
    val_dataset = GoogleLandmark(data_dir='/data/google-landmark/org/train',
                                 data_csv='/data/google-landmark/csv/val-clean-1000.csv',
                                 transform=val_transform,
                                 idx_to_class=train_dataset.idx_to_class)


    # return train/val dataloader
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = train_batch_size,
                                               shuffle = False,
                                               num_workers = workers)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                             batch_size = val_batch_size,
                                             shuffle = False,
                                             num_workers = workers)

    return train_loader, val_loader, train_dataset.num_classes # also return ncls



