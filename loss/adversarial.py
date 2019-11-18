import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable


def discriminator_loss(discriminator: nn.Module, fake: torch.Tensor, real: torch.Tensor):
    # loss of Discriminator
    d_fake = discriminator(fake.detach())
    d_real = discriminator(real.detach())

    label_fake = d_fake.data.new(d_fake.size()).fill_(0)
    label_real = d_real.data.new(d_real.size()).fill_(1)
    loss_d = F.mse_loss(d_fake, label_fake) + F.mse_loss(d_real, label_real)

    return loss_d


def cycle_loss(generator_ab: nn.Module, generator_ba: nn.Module, real: torch.Tensor):
    # loss of 2 generators
    fake = generator_ba(generator_ab(real))
    loss_cycle = F.mse_loss(fake, real)
    return loss_cycle


def generator_discriminator_loss(generator: nn.Module, discriminator: nn.Module, input: torch.Tensor):
    # loss for generator and discriminator
    fake = generator(input)
    d_fake = discriminator(fake)
    label_d_fake = d_fake.data.new(d_fake.size()).fill_(1)

    # print(d_fake.shape, label_d_fake.shape)

    loss_g = F.mse_loss(d_fake, label_d_fake)
    return loss_g

