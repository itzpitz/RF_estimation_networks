# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys, os
sys.path.append('micro_optimisation')
import logging
import torch
import torch.nn as nn
import torchvision as tv

from torch.utils.data import DataLoader, random_split
from customSet import CustomSet
from utils import reward_accuracy

import numpy as np
from scipy import stats

logger = logging.getLogger('nni')
logger.propagate = False


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


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    batch_size = 128

    norms = {
        'CLx': [2.2436, 1.0114],
        'CLy': [2.2448, 1.0112],
        'angle': [22.5027, 12.9941],
        'var': [0.1271, 0.0683],
    }

    n_digits = {
            'CLx': 4,
            'CLy': 4,
            'angle': 2,
            'var': 2
            }

    transform = tv.transforms.Compose([
        tv.transforms.Normalize((0.01,), (0.0015,))
    ])

    for output in ['CLx', 'CLy']:
        print(output)

        mean = norms[output][0]
        std = norms[output][1]

        full_dataset = CustomSet(image_dir='eval_set',
                                 mean=mean,
                                 std=std,
                                 norm=True,
                                 output=output,
                                 transform=transform,
                                 seed=123,
                                 )

        full_dataset = DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=False)

        model = Net()
        model = nn.DataParallel(model).to(device)

        # apply_fixed_architecture(model, os.path.join("checkpoints_{0}".format(output), 'epoch_149.json'))

        model.load_state_dict(torch.load(os.path.join('trained_models', 'trained_model_' + output +
                                                                   '_old_architecture.model'),
                              map_location=device))

        output_predictions = []
        output_true = []
        acc_list = []
        abs_err_list = []

        model.eval()
        mae = nn.L1Loss()
        with torch.no_grad():
            for i, batch in enumerate(full_dataset):
                x, y = batch[0], batch[1]
                x, y = x.to(device), y.to(device)
                # forward pass
                pred = model(x)
                # now do the inverse normalization
                pred = pred * std + mean
                y = y * std + mean
                # do the rounding
                pred = torch.round(pred * 10**n_digits[output]) / (10**n_digits[output])
                y = torch.round(y * 10**n_digits[output]) / (10**n_digits[output])
                
                acc = reward_accuracy(pred, y)
                abs_err = mae(pred, y)
                
                acc_list.extend([acc for _ in range(y.size()[0])])
                abs_err_list.extend([abs_err.item() for _ in range(y.size()[0])])

                output_predictions.extend(torch.reshape(pred, (-1,)).tolist())
                output_true.extend(torch.reshape(y, (-1,)).tolist())

        res = stats.linregress(output_predictions, output_true)
        print('Output', output)
        print('Average accuracy', np.mean(acc_list))
        print('Mean absolute error', np.mean(abs_err_list))
        print(f"R-squared: {res.rvalue ** 2:.6f}")
        print('Nr. of images:', len(output_predictions))

        with open('cnn_predictions_old_architecture_' + output + '.txt', 'w+') as file:
            file.write('Ground truth; Network Prediction\n')
            for idx, y in enumerate(output_true):
                file.write('{:5f}'.format(y))
                file.write(';')
                file.write('{:5f}'.format(output_predictions[idx]))
                file.write('\n')


if __name__ == "__main__":
    main()
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)
