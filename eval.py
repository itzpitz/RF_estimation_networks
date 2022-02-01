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
from micro import MicroNetwork
from utils import reward_accuracy

from nni.nas.pytorch.fixed import apply_fixed_architecture

import numpy as np
from scipy import stats
import json

logger = logging.getLogger('nni')
logger.propagate = False

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    batch_size = 128

    with open('stats.json', 'r') as file:
        norms = json.load(file)

    n_digits = {
            'CLx': 4,
            'CLy': 4,
            'angle': 2,
            'var': 2
            }

    transform = tv.transforms.Compose([
        tv.transforms.Normalize((norms['transform']['mean'],), (norms['transform']['std'],))
    ])

    for output in ['CLx', 'CLy', 'angle', 'var']:
        print(output)

        mean = norms[output]['mean']
        std = norms[output]['std']

        full_dataset = CustomSet(image_dir='eval_set',
                                 mean=mean,
                                 std=std,
                                 norm=True,
                                 output=output,
                                 transform=transform,
                                 seed=123,
                                 )

        full_dataset = DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=False)

        model = MicroNetwork(num_layers=6, out_channels=24, num_nodes=5, dropout_rate=0.25, use_aux_heads=False)
        model = nn.DataParallel(model, device_ids=[0]).to(device)
        
        search_epoch = 310
        if output == 'CLy':
            apply_fixed_architecture(model, os.path.join("checkpoints_{0}_old_search".format(output), 'epoch_%s.json' %(search_epoch - 1)))
        else:
            apply_fixed_architecture(model, os.path.join("checkpoints_{0}".format(output), 'epoch_%s.json' %(search_epoch - 1)))

        if search_epoch == 150:
            model.load_state_dict(torch.load(os.path.join('trained_models', 'trained_model_' + output +
                                                                                   '.model'),
                                  map_location=device))
        else:
            model.load_state_dict(torch.load(os.path.join('trained_models', 'trained_model_' + output +
                                                                   '_epoch%s.model' %search_epoch),
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

        with open(os.path.join('predictions', 'cnn_predictions_' + output + '.txt'), 'w+') as file:
            file.write('Ground truth; Network Prediction\n')
            for idx, y in enumerate(output_true):
                file.write('{:5f}'.format(y))
                file.write(';')
                file.write('{:5f}'.format(output_predictions[idx]))
                file.write('\n')


if __name__ == "__main__":
    main()
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)
