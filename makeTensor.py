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


img_dir = '../FEM/training'
out_dir = 'eval_set'
root = os.path.dirname(os.path.realpath(__file__))
# root = os.path.join(root, img_dir)
for file in os.listdir(os.path.join(root, img_dir)):
    if file.split('.')[-1] == 'json':
        print(file)
        with open(os.path.join(os.path.join(root, img_dir), file), 'r', encoding='utf-8') as f:
            x = torch.tensor(json.loads(f.read()))
        x = torch.reshape(x, (1, x.size()[0], x.size()[1])).type(dtype=torch.float32)
        torch.save(x, os.path.join(os.path.join(root, out_dir), file.split('.')[0] + '.pt'))
