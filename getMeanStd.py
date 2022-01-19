# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
sys.path.append('micro_optimisation')
import torch
import torch.nn as nn
import torchvision as tv

from torch.utils.data import DataLoader, random_split
from micro import MicroNetwork
from customSet import CustomSet
from utils import reward_accuracy

from nni.nas.pytorch.fixed import apply_fixed_architecture

import numpy as np

import json


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    label_sum, label_squared_sum = 0, 0
    for data, label in dataloader:
        # Mean over batch, height and width, but not over the channels
        size = data.size()[0]
        channels_sum += torch.mean(data, dim=[0, 2, 3]) * size
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3]) * size
        label_sum += torch.mean(label) * size
        label_squared_sum += torch.mean(label ** 2) * size

        num_batches += size
        print(num_batches)

    mean = channels_sum / num_batches

    mean_label = label_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    std_label = (label_squared_sum / num_batches - mean_label ** 2) ** 0.5

    return mean.item(), std.item(), mean_label.item(), std_label.item()


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    batch_size = 5120
    stats_dict = {}
    for output in ['CLx', 'CLy', 'var', 'angle']:
        print(output)
        full_dataset = CustomSet(image_dir='training_torch',
                                 norm=False,
                                 output=output,
                                 transform=None,
                                 seed=123,
                                 )

        data = DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=False)
        mean, std, mean_label, std_label = get_mean_and_std(data)
        stats_dict[output] = {
            'mean': mean_label,
            'std': std_label
        }

    stats_dict['transform'] = {'mean': mean, 'std': std}
    with open('stats.json', 'w+') as file:
        json.dump(stats_dict, file, indent=4)
    print(stats_dict)



if __name__ == "__main__":
    main()
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)
