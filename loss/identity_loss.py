import torch
import torch.nn.functional as F


def identity_loss(clean_image: torch.Tensor, generator: torch.nn.Module):
    fake_clean_image = generator(clean_image)
    loss_identity = F.mse_loss(clean_image, fake_clean_image)
    return loss_identity