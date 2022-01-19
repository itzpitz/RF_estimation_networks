# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchvision as tv

# import datasets
from torch.utils.data import random_split
from customSet import CustomSet
from micro import MicroNetwork
from nni.algorithms.nas.pytorch import enas
from nni.nas.pytorch.callbacks import (ArchitectureCheckpoint,
                                       LRSchedulerCallback)
from utils import accuracy, reward_accuracy

import json

logger = logging.getLogger('nni')


if __name__ == "__main__":
    parser = ArgumentParser("enas")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--train_split", default=0.8, type=float)
    parser.add_argument("--log_frequency", default=10, type=int)
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs (default: macro 310, micro 150)")
    parser.add_argument("--visualization", default=False, action="store_true")
    parser.add_argument("--v1", default=True, action="store_true")
    parser.add_argument("--output", default='CLx', type=str)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    output = 'CLy'
    mean = 2.2448
    std = 1.0112
    norm = True

    transform = tv.transforms.Compose([
        tv.transforms.Normalize((0.01,), (0.0015,))
    ])
    full_dataset = CustomSet(image_dir='training_torch',
                             mean=mean,
                             std=std,
                             norm=norm,
                             output=output,
                             transform=transform,
                             seed=123,
                             num_images=75000)

    train_len = int(len(full_dataset) * args.train_split)
    val_len = len(full_dataset) - train_len
    dataset_train, dataset_valid = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    # train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
    # insert code here

    # mutator = None
    # ctrl_kwargs = {}
    # if args.search_for == "macro":
    #     model = GeneralNetwork()
    #     num_epochs = args.epochs or 310
    # elif args.search_for == "micro":
    model = MicroNetwork(num_layers=6, out_channels=24, num_nodes=5, dropout_rate=0.25, use_aux_heads=False)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids=[0, 2])
    num_epochs = args.epochs or 150
    if args.v1:
        mutator = enas.EnasMutator(model, tanh_constant=1.1, cell_exit_extra_step=True)
    else:
        ctrl_kwargs = {"tanh_constant": 1.1}
    # else:
    #     raise AssertionError

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)

    if args.v1:
        trainer = enas.EnasTrainer(model,
                                   loss=criterion,
                                   metrics=accuracy,
                                   reward_function=reward_accuracy,
                                   optimizer=optimizer,
                                   callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints_{0}".format(output))],
                                   batch_size=args.batch_size,
                                   num_epochs=num_epochs,
                                   dataset_train=dataset_train,
                                   dataset_valid=dataset_valid,
                                   log_frequency=args.log_frequency,
                                   mutator=mutator)
        if args.visualization:
            trainer.enable_visualization()
        trainer.train()
        trainer.export(file="optimised_{0}.json".format(output))
    else:
        from nni.retiarii.oneshot.pytorch.enas import EnasTrainer
        trainer = EnasTrainer(model,
                              loss=criterion,
                              metrics=accuracy,
                              reward_function=reward_accuracy,
                              optimizer=optimizer,
                              batch_size=args.batch_size,
                              num_epochs=num_epochs,
                              dataset=full_dataset,
                              log_frequency=args.log_frequency,
                              ctrl_kwargs=ctrl_kwargs,
                              device=device
                              )
        # if args.visualization:
        #     trainer.enable_visualization()
        trainer.fit()
        trainer.export()
