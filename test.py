import argparse
import torch
from torch import nn
import torchvision
import os
import timeit
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from torchvision.datasets.cityscapes import Cityscapes
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm

from models import EDSR, Generator_sr, Generator_lr


def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # config
    parser.add_argument('-d', '--data-path', default='/data/data/DIV2K/unsupervised/lr', type=str)
    parser.add_argument('-g', '--gpu', default='0', type=str)
    parser.add_argument('-w', '--weights-dir', default='output-0/', type=str)
    parser.add_argument('-w', '--output-dir', default='output-0-predict/', type=str)

    args = parser.parse_args()

    print('-' * 20)
    for key in args.__dict__:
        print(key, '=', args.__dict__[key])
    print('-' * 20)

    return args


def resolv_sr(G_1: nn.Module, SR: nn.Module, image: Image):
    image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0).cuda()
    sr_image_tensor = SR(G_1(image_tensor)).cpu()
    sr_image = torchvision.transforms.functional.to_pil_image(sr_image_tensor[0])
    return sr_image


def resolv_deonoise(G_1: nn.Module, image: Image):
    image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0).cuda()
    clean_image_tensor = G_1(image_tensor).cpu()
    clean_image = torchvision.transforms.functional.to_pil_image(clean_image_tensor[0])
    return clean_image


def main(args):
    G_1 = Generator_lr(in_channels=3)
    SR = EDSR(n_colors=3)

    # load pretrained model
    G_1.load_state_dict(torch.load(os.path.join(args.weights_dir, 'final_weights_G_1.pkl')))
    SR.load_state_dict(torch.load(os.path.join(args.weights_dir, 'final_weights_SR.pkl')))

    G_1.cuda()
    G_1.eval()
    SR.cuda()
    SR.eval()

    # predict
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'clean'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'SR'), exist_ok=True)
    for image_name in tqdm(os.listdir(args.data_path)):
        # read file
        image = Image.open(os.path.join(args.data_path, image_name))
        # denoise
        clean_image = resolv_deonoise(G_1, image)
        clean_image.save(os.path.join(args.output_dir, 'clean', image_name))
        # SR
        sr_image = resolv_sr(G_1, SR, image)
        sr_image.save(os.path.join(args.output_dir, 'SR', image_name))


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main(args)