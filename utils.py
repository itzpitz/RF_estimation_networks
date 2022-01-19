# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

# def accuracy(output, target, topk=(1,)):
#     """ Computes the precision@k for the specified values of k """
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     # one-hot case
#     if target.ndimension() > 1:
#         target = target.max(1)[1]
#
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = dict()
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
#     return res
#
#
# def reward_accuracy(output, target, topk=(1,)):
#     batch_size = target.size(0)
#     _, predicted = torch.max(output.data, 1)
#     return (predicted == target).sum().item() / batch_size


def accuracy(output, target):
    """ Computes the precision@k for the specified values of k """
    acc = reward_accuracy(output, target)
    return {'acc': acc}


def reward_accuracy(output, target, pct_close=.1):
    # pure Tensor, efficient version
    n_items = target.size(0)
    # X = T.Tensor(data_x)
    # Y = T.Tensor(data_y)  # actual as [102] Tensor
    #
    # oupt = model(X)       # predicted as [102,1] Tensor
    # pred = oupt.view(n_items)  # predicted as [102]

    n_correct = torch.sum((torch.abs(output - target) < torch.abs(pct_close * target)))
    acc = (n_correct.item() * 100.0 / n_items)  # scalar
    return acc


# def reward_accuracy(x, y):
#     criterion = nn.MSELoss()
#     acc = criterion(x, y)
#     return acc