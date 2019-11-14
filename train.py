import argparse
import torch
import torchvision
import os
import timeit
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from torchvision.datasets.cityscapes import Cityscapes
from torchvision.transforms.functional import to_tensor

from data import DIV2KDataset
from models import Discriminator_sr, Discriminator_lr, EDSR, Generator_sr, Generator_lr


def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # config
    parser.add_argument('-d', '--data-path', default='/data/data/DIV2K/train_dataset', type=str)
    parser.add_argument('-g', '--gpu', default='0', type=str)
    parser.add_argument('-l', '--log-dir', default='output-0/', type=str)
    # Train Setting
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    args = parser.parse_args()

    print('-' * 20)
    for key in args.__dict__:
        print(key, '=', args.__dict__[key])
    print('-' * 20)

    return args


