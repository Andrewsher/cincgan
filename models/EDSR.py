from models import common

import torch.nn as nn
import torch.nn.functional as F


def make_model(parent=False):
    return EDSR()


class EDSR(nn.Module):
    def __init__(self, conv=common.default_conv, n_resblock=8, n_filters=64, scale=4, rgb_range=255, n_colors=1, res_scale=1):
        super(EDSR, self).__init__()

        # n_resblock = args.n_resblocks
        # n_feats = args.n_feats
        kernel_size = 3
        # scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(rgb_range, rgb_mean, -1)  # rgb_std)

        # define head module
        m_head = [conv(n_colors, n_filters, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_filters, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_filters, n_filters, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_filters, act=False),
            conv(n_filters, n_colors, kernel_size)
        ]

        # self.add_mean = common.MeanShift(rgb_range, rgb_mean, 1)  # rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = x * 255
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)
        # x = x / 255
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == '__main__':
    model = EDSR()
    print(model)
    print("# of parameter:", sum(param.numel() for param in model.parameters()))
    import numpy as np

    input_map = np.zeros((1, 16, 16))
    # input_map = np.reshape(input_map, (1, 64, 64))
    import torchvision, torch

    # input_map = torchvision.transforms.functional.to_tensor(input_map)
    input_maps = torch.as_tensor(data=[input_map, input_map, input_map, input_map], dtype=torch.float)
    print(input_maps.shape)
    output_map = model(input_maps)
    print(output_map.shape)