import torch
import torch.nn.functional as F


def identity_loss(clean_image: torch.Tensor, generator: torch.nn.Module):
    fake_clean_image = generator(clean_image)
    loss_identity = F.mse_loss(clean_image, fake_clean_image)
    return loss_identity


def identity_loss_sr(clean_image_lr: torch.Tensor, clean_image_hr: torch.Tensor, generator: torch.nn.Module):
    clean_image_sr = generator(clean_image_lr)
    loss_identity = F.mse_loss(clean_image_sr, clean_image_hr)
    return loss_identity
