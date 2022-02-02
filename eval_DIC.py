# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys, os
sys.path.append('micro_optimisation')
import logging
import torch
import torch.nn as nn
import torchvision as tv

from torch.utils.data import DataLoader, random_split
from customSetDIC import CustomSet
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

        full_dataset = CustomSet(image_dir='DIC_torch',
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
        output_names = []

        model.eval()
        mae = nn.L1Loss()
        with torch.no_grad():
            for i, batch in enumerate(full_dataset):
                x, names = batch[0], batch[1]
                x  = x.to(device)
                # forward pass
                pred = model(x)
                # now do the inverse normalization
                pred = pred * std + mean
                # do the rounding
                pred = torch.round(pred * 10**n_digits[output]) / (10**n_digits[output])
                
                output_predictions.extend(torch.reshape(pred, (-1,)).tolist())
                output_names.extend(names)

        print(output_predictions)
        with open(os.path.join('predictions', 'DIC_' + output + '.txt'), 'w+') as file:
            file.write('Ground truth; Network Prediction\n')
            for idx, y in enumerate(output_predictions):
                file.write(output_names[idx])
                file.write(';{:5f}'.format(y))
                file.write('\n')


if __name__ == "__main__":
    main()
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)
