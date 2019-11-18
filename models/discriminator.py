from torch import nn
import torch.nn.functional as F


class Discriminator_lr(nn.Module):
    def __init__(self, in_channels=1, in_h=16, in_w=16):
        super(Discriminator_lr, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1)

        self.linear1 = nn.Linear(in_features=1*in_h*in_w, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=1)

    def forward(self, x):
        y = F.leaky_relu(self.conv1(x), inplace=True)
        y = F.leaky_relu(self.bn1(self.conv2(y)), inplace=True)
        y = F.leaky_relu(self.bn2(self.conv3(y)), inplace=True)
        y = F.leaky_relu(self.bn3(self.conv4(y)), inplace=True)
        y = F.leaky_relu(self.conv5(y), inplace=True)

        # y = y.flatten(1, -1)
        # y = nn.Linear(in_features=y.shape[-1], out_features=1024)
        # y.shape[-1]

        # y = F.relu(self.linear1(y.view(y.size(0), -1)), inplace=True)
        y = F.leaky_relu(self.linear1(y.flatten(1, -1)), inplace=True)
        y = F.sigmoid(self.linear2(y))

        return y


class Discriminator_sr(nn.Module):
    def __init__(self, in_channels=1, in_h=64, in_w=64):
        super(Discriminator_sr, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1)

        self.linear1 = nn.Linear(in_features=1*in_h*in_w//16, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        y = F.leaky_relu(self.conv1(x), inplace=True)
        y = F.leaky_relu(self.bn1(self.conv2(y)), inplace=True)
        y = F.leaky_relu(self.bn2(self.conv3(y)), inplace=True)
        y = F.leaky_relu(self.bn3(self.conv4(y)), inplace=True)
        y = F.leaky_relu(self.conv5(y), inplace=True)

        y = y.flatten(1, -1)
        y = F.leaky_relu(self.linear1(y), inplace=True)
        # y = nn.Linear(in_features=y.shape[-1], out_features=1024)
        # y.shape[-1]

        # y = F.relu(self.linear1(y.view(y.size(0), -1)), inplace=True)
        y = F.sigmoid(self.linear2(y))

        return y



if __name__ == '__main__':
    model = Discriminator_sr()
    print(model)
    # for parameter in model.parameters():
    #     print(parameter)
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