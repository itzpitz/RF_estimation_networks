# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import torch
import torch.nn as nn

import torchvision as tv

from torch.utils.data import Dataset

import os
import pandas as pd
import json
import random


class CustomSet(Dataset):
    def __init__(self,
                 image_dir,
                 cl_min=0.5,
                 cl_max=4.,
                 angle_max=45.,
                 var_min=0.01,
                 var_max=0.25,
                 minmaxtransform=True,
                 # output can be one of CLx, CLy, angle
                 output='CLx',
                 transform=tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize((0.5, ), (0.5, ))
                    ]),
                 seed=123,
                 num_images=None
                 ):
        self.seed = seed
        self.num_images = num_images
        self.img_dir = image_dir
        self.img_names = self.get_names()
        self.transform = transform
        # self.to_tensor = tv.transforms.ToTensor()
        # self.to_pil = tv.transforms.ToPILImage()
        self.output = output
        self.cl_min = cl_min
        self.cl_max = cl_max
        self.angle_max = angle_max
        self.var_min = var_min
        self.var_max = var_max
        self.minmaxtransform = minmaxtransform

    def get_names(self):
        """ returns a dataframe containing the names of all images in the folder """
        root = os.path.dirname(os.path.realpath(__file__))
        root = os.path.join(root, self.img_dir)
        filenames = []
        if self.num_images is None:
            for file in os.listdir(root):
                if file.split('.')[-1].lower() == 'pt':
                    filenames.append(file)
        else:
            for file in random.sample(os.listdir(root), self.num_images):
                if file.split('.')[-1].lower() == 'pt':
                    filenames.append(file)

        return pd.DataFrame(filenames)

    def load_json(self, name):
        root = os.path.dirname(os.path.realpath(__file__))
        root = os.path.join(root, self.img_dir)
        return torch.load(os.path.join(root, name))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        X = self.load_json(self.img_names.iloc[index, -1])
        if self.transform is not None:
            X = self.transform(X)
        temp = self.img_names.iloc[index, -1].split('.')[0]
        clx = float(temp.split('_')[0].replace('CL', '').replace('d', '.'))
        cly = float(temp.split('_')[1].replace('d', '.'))
        angle = float(temp.split('_')[2].replace('d', '.'))
        var = float(temp.split('_')[3].replace('d', '.'))
        # print(clx, cly, angle)
        # normalize the correlation length
        if self.minmaxtransform:
            clx = (clx - self.cl_min) / (self.cl_max - self.cl_min)
            cly = (cly - self.cl_min) / (self.cl_max - self.cl_min)
            angle /= self.angle_max
            var = (var - self.var_min) / (self.var_max - self.var_min)

        if self.output == 'CLx':
            Y = clx
        elif self.output == 'CLy':
            Y = cly
        elif self.output == 'angle':
            Y = angle
        elif self.output == 'var':
            Y = var
        else:
            raise NameError
        # sample = {'X': X, 'Y': Y}
        return X, torch.tensor(Y, dtype=torch.float32).reshape(-1)
