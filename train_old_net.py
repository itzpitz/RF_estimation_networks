# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
import os
import torch
import torch.nn as nn
import torchvision as tv

from torch.utils.data import DataLoader, random_split
from customSet import CustomSet
from utils import reward_accuracy
from argparse import ArgumentParser
import json

import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # w_out = (w_in - filter_size +2*padding)/stride + 1
        self.layer1 = nn.Sequential(
            # padding = (kernel_size - 1)/2 to get same size as before
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            # non trainable batch normalization
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
        )
        # (60,60,16)
        self.layer2 = nn.Sequential(
            # padding = (kernel_size - 1)/2 to get same size as before
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # non trainable batch normalization
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (30,30,32)
        self.layer3 = nn.Sequential(
            # padding = (kernel_size - 1)/2 to get same size as before
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # (30,30,64)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # (30,30,64)
        self.layer5 = nn.Sequential(
            # padding = (kernel_size - 1)/2 to get same size as before
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # non trainable batch normalization
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        #         (15,15,128)
        self.layer6 = nn.Sequential(
            nn.Linear(15*15*128, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.layer7 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.out = nn.Linear(1024, 1)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        out = self.out(x)
        return out


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
            torch.save(model.state_dict(), os.path.join('trained_models', 'trained_model_{0}_old_arch.model'.format(output))

        if scheduler is not None:
            scheduler.step()

        print('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, Validation '
              'MAE: {:.4f}'
              .format(epoch + 1, num_epochs, loss_list[-1], val_loss[-1], val_acc_epoch, val_mae_epoch))

    with open(os.path.join('train_data', 'training_data_{0}_old_arch.txt'.format(output), 'w+') as f:
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

    # python train.py --batch_size 128 --epochs 10 --gpu 1 --output CLy --nr_images 10000

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

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

    model = Net()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if args.gpu is None:
        model = nn.DataParallel(model).to(device)
    else:
        model = nn.DataParallel(model, device_ids=[args.gpu]).to(device)

    # apply_fixed_architecture(model, os.path.join("checkpoints_{0}".format(output), 'epoch_149.json'))

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
    torch.save(state, 'state_continued_training_{0}_checkpoint'.format(output))


if __name__ == "__main__":
    main()
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)
