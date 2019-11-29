from PIL import Image
import argparse
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='crop')
    # config
    parser.add_argument('-h', '--hr-path', default='/data/data/DIV2K/hr_train', type=str)
    parser.add_argument('-l', '--lr-path', default='/data/data/DIV2K/lr_train', type=str)
    parser.add_argument('-t', '--target-dir', default='/data/data/DIV2K/train_dataset', type=str)
    parser.add_argument('--crop-size', default=64, type=int)
    parser.add_argument('--crop-step', default=32, type=int)
    parser.add_argument('--scale', default=4, type=int)

    args = parser.parse_args()

    for key in args.__dict__.keys():
        print(key, '=', args.__dict__[key])

    return args


def crop(args):
    print('start cropping')
    # make dir
    os.makedirs(args.target_dir, exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'lr'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'hr'), exist_ok=True)

    # open image
    assert len(os.listdir(args.lr_path)) == os.listdir(args.hr_path)
    for image_name in tqdm(os.listdir(args.hr_path)):
        image_hr = Image.open(os.path.join(args.hr_path, image_name))
        image_lr = Image.open(os.path.join(args.lr_path, image_name.split('.')[0] + 'x4d.png'))
        # crop
        for cx in range(0, image_hr.size[0] - args.crop_size, args.crop_step):
            for cy in range(0, image_hr.size[1] - args.crop_size, args.crop_step):
                current_image_name = '{}_{}_{}.png'.format(image_name.split('.')[0], str(cx), str(cy))
                # hr
                current_image_hr = image_hr.crop((cx,
                                                  cy,
                                                  cx + args.crop_size,
                                                  cy + args.crop_size))
                current_image_hr.save(os.path.join(args.target_dir, 'hr', current_image_name))
                # lr
                current_image_lr = image_lr.crop((cx // args.scale,
                                                  cy // args.scale,
                                                  (cx // args.scale) + (args.crop_size // args.scale),
                                                  (cy // args.scale) + (args.crop_size // args.scale)))
                current_image_lr.save(os.path.join(args.target_dir, 'lr', current_image_name))

    print('cropping done')


if __name__ == '__main__':
    args = parse_args()
    crop(args)
