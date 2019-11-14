import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable


def adversarial_loss_lr(discriminator: nn.Module, fake: torch.Tensor, real: torch.Tensor):
    fake_detach = fake.detach()
    d_fake = discriminator(fake_detach)
    d_real = discriminator(real)

    label_fake = d_fake.data.new(d_fake.size()).fill_(0)
    label_real = d_real.data.new(d_real.size()).fill_(1)
    loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake) + \
             F.binary_cross_entropy_with_logits(d_real, label_real)

