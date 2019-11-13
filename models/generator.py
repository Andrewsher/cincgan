from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x), inplace=True)
        y = F.relu(self.conv2(y), inplace=True)
        y = x + y
        return y


class Generator_lr(nn.Module):
    def __init__(self, in_channels=1):
        super(Generator_lr, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.block1 = Block(in_channels=64, out_channels=64)
        self.block2 = Block(in_channels=64, out_channels=64)
        self.block3 = Block(in_channels=64, out_channels=64)
        self.block4 = Block(in_channels=64, out_channels=64)
        self.block5 = Block(in_channels=64, out_channels=64)
        self.block6 = Block(in_channels=64, out_channels=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=7, padding=3)

    def forward(self, x):
        y = F.relu(self.conv1(x), inplace=True)
        y = F.relu(self.conv2(y), inplace=True)
        y = F.relu(self.conv3(y), inplace=True)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.block6(y)
        y = F.relu(self.conv4(y), inplace=True)
        y = F.relu(self.conv5(y), inplace=True)
        y = F.relu(self.conv6(y), inplace=True)
        return y


class Generator_sr(nn.Module):
    def __init__(self, in_channels=1):
        super(Generator_sr, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.block1 = Block(in_channels=64, out_channels=64)
        self.block2 = Block(in_channels=64, out_channels=64)
        self.block3 = Block(in_channels=64, out_channels=64)
        self.block4 = Block(in_channels=64, out_channels=64)
        self.block5 = Block(in_channels=64, out_channels=64)
        self.block6 = Block(in_channels=64, out_channels=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=7, padding=3)

    def forward(self, x):
        y = F.relu(self.conv1(x), inplace=True)
        y = F.relu(self.conv2(y), inplace=True)
        y = F.relu(self.conv3(y), inplace=True)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.block6(y)
        y = F.relu(self.conv4(y), inplace=True)
        y = F.relu(self.conv5(y), inplace=True)
        y = F.relu(self.conv6(y), inplace=True)
        return y


if __name__ == '__main__':
    model = Generator_sr()
    print(model)
    print("# of parameter:", sum(param.numel() for param in model.parameters()))
    import numpy as np
    input_map = np.zeros((1, 64, 64))
    # input_map = np.reshape(input_map, (1, 64, 64))
    import torchvision, torch
    # input_map = torchvision.transforms.functional.to_tensor(input_map)
    input_maps = torch.as_tensor(data=[input_map, input_map, input_map, input_map], dtype=torch.float)
    print(input_maps.shape)
    output_map = model(input_maps)
    print(output_map.shape)
    