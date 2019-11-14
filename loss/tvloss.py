import torch
from torch import nn


def tensor_size(t: torch.Tensor):
    return t.size()[1] * t.size()[2] * t.size()[3]


def tvloss(input: torch.Tensor, generator: nn.Module):
    x = generator(input)
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = tensor_size(x[:, :, 1:, :])
    count_w = tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
