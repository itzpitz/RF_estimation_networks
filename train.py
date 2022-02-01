# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
sys.path.append('micro_optimisation')
import os
import torch
import torch.nn as nn
import torchvision as tv

from torch.utils.data import DataLoader, random_split
from micro import MicroNetwork
from customSet import CustomSet
from utils import reward_accuracy

from nni.nas.pytorch.fixed import apply_fixed_architecture

import numpy as np
from argparse import ArgumentParser
import json

torch.multiprocessing.set_sharing_strategy('file_system')


def train(model, num_epochs, training_set, validation_set, criterion, optimizer, device, output, scheduler=None,
          log_frequency=100, norm=False, mean=0, std=1):

    total_step = len(training_set)
    loss_list = []
    val_loss = []
    val_accuracy = []
    val_mae = []
    best_loss = np.inf
    for epoch in range(num_epochs):
        loss_epoch = []
        model.train()
        for i, batch in enumerate(training_set):
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            # forward pass
            pred = model(x)
            loss = criterion(pred, y)
            loss_epoch.append(loss.item())

            # Backdrop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % log_frequency == 0:
                print('Epoch [{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, i+1, total_step, loss_epoch[-1]))

        loss_list.append(np.mean(loss_epoch))

        val_loss_epoch, val_acc_epoch, val_mae_epoch = test(model, validation_set, criterion, device, norm=norm,
                                                            mean=mean, std=std)
        val_loss.append(val_loss_epoch)
        val_mae.append(val_mae_epoch)
        val_accuracy.append(val_acc_epoch)

        if val_loss_epoch < best_loss:
            print('Validation loss improved from:', best_loss, 'to', val_loss_epoch)
            best_loss = val_loss_epoch
            torch.save(model.state_dict(), os.path.join('trained_models',
                                                        'trained_model_{0}.model'.format(output)))

        if scheduler is not None:
            scheduler.step()

        print('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, Validation '
              'MAE: {:.4f}'
              .format(epoch + 1, num_epochs, loss_list[-1], val_loss[-1], val_acc_epoch, val_mae_epoch))

    with open(os.path.join('train_data', 'training_data_{0}.txt'.format(output)), 'w+') as f:
        f.write('epoch; loss, validation loss; validation accuracy; validation mae\n')
        for i in range(num_epochs):
            f.write('{}; {:.4f}; {:.4f}; {:.4f}; {:.4f}\n'.format(i+1, loss_list[i], val_loss[i],
                                                                  val_accuracy[i], val_mae[i]))

    #     nni.report_intermediate_result(val_loss[-1])
    # nni.report_final_result(best_loss)


def test(model, validation_set, criterion, device, norm=False, mean=0, std=1):
    val_loss = []
    val_accuracy = []
    val_mae = []
    mae = nn.L1Loss()
    with torch.no_grad():
        for i, batch in enumerate(validation_set):
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            # forward pass
            model.eval()
            pred = model(x)
            loss = criterion(pred, y)
            # calculate accuracy and mae on true values
            if norm:
                pred = pred * std + mean
                y = y * std + mean
            acc = reward_accuracy(pred, y)
            val_loss.append(loss.item())
            val_accuracy.append(acc)
            val_mae.append(mae(pred, y).item())
    return np.mean(val_loss), np.mean(val_accuracy), np.mean(val_mae)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = ArgumentParser("train")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--log_frequency", default=10, type=int)
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs (default: macro 310, micro 150)")
    parser.add_argument("--output", default='CLx', type=str)
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--nr_images", default=None, type=int)
    args = parser.parse_args()

    # test first!!
    # python train.py --batch_size 128 --epochs 10 --gpu 1 --output CLy --nr_images 10000
    # needs folders checkpoints_output, continued_training, train_data, trained_models

    if torch.cuda.is_available():
        if args.gpu is None:
            device = torch.device("cuda:0")
        else:
            print("Running on GPU", args.gpu)
            device = torch.device("cuda:%s" % args.gpu)
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    with open('stats.json', 'r') as file:
        stats = json.load(file)

    mean = stats[args.output]['mean']
    std = stats[args.output]['std']
    norm = True

    batch_size = args.batch_size
    log_frequency = args.log_frequency
    num_epochs = args.epochs
    output = args.output
    learning_rate = 1e-3
    train_split = 0.8

    transform = tv.transforms.Compose([
        tv.transforms.Normalize((stats['transform']['mean'],), (stats['transform']['std'],))
    ])

    full_dataset = CustomSet(image_dir='training_torch',
                             mean=mean,
                             std=std,
                             norm=norm,
                             output=output,
                             transform=transform,
                             seed=123,
                             num_images=args.nr_images
                             )

    train_len = int(len(full_dataset) * train_split)
    val_len = len(full_dataset) - train_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    #
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    model = MicroNetwork(num_layers=6, out_channels=24, num_nodes=5, dropout_rate=0.25, use_aux_heads=False)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if args.gpu is None:
        model = nn.DataParallel(model).to(device)
    else:
        model = nn.DataParallel(model, device_ids=[args.gpu]).to(device)

    apply_fixed_architecture(model, os.path.join("checkpoints_{0}".format(output), "epoch_309.json"))

    print('Number of trainable parameters: {0}'.format(count_parameters(model)))

    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, threshold=1e-3,
    #                                                        verbose=True)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    train(model, num_epochs, train_loader, val_loader, criterion, optimizer, device, output, scheduler=scheduler,
          log_frequency=log_frequency, norm=norm, mean=mean, std=std)
    # torch.save(model.state_dict(), 'trained_model_CLx')

    #     save for continued training
    state = {'epoch': num_epochs + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), }
    torch.save(state, os.path.join('continued_training', 'state_continued_training_{0}.model'.format(output)))


if __name__ == "__main__":
    main()
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)
