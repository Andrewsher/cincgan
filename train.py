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

from data import DIV2KDataset
from models import Discriminator_sr, Discriminator_lr, EDSR, Generator_sr, Generator_lr
from loss import generator_discriminator_loss, discriminator_loss, cycle_loss, identity_loss, identity_loss_sr, tvloss


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


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.makedirs(args.log_dir, exist_ok=True)

    # create models
    G_1 = Generator_lr(in_channels=3)
    G_2 = Generator_lr(in_channels=3)
    D_1 = Discriminator_lr(in_channels=3, in_h=16, in_w=16)
    SR = EDSR(n_colors=3)
    G_3 = Generator_sr(in_channels=3)
    D_2 = Discriminator_sr(in_channels=3, in_h=64, in_w=64)

    for model in [G_1, G_2, D_1, SR, G_3, D_2]:
        model.cuda()
        model.train()

    # create optimizors
    optim = {
        'G_1': torch.optim.SGD(params=filter(lambda p: p.requires_grad, G_1.parameters()), lr=args.lr),
        'G_2': torch.optim.SGD(params=filter(lambda p: p.requires_grad, G_2.parameters()), lr=args.lr),
        'D_1': torch.optim.SGD(params=filter(lambda p: p.requires_grad, D_1.parameters()), lr=args.lr),
        'SR': torch.optim.SGD(params=filter(lambda p: p.requires_grad, SR.parameters()), lr=args.lr),
        'G_3': torch.optim.SGD(params=filter(lambda p: p.requires_grad, G_3.parameters()), lr=args.lr),
        'D_2': torch.optim.SGD(params=filter(lambda p: p.requires_grad, D_2.parameters()), lr=args.lr)
    }
    for key in optim.keys():
        optim[key].zero_grad()

    # get dataloader
    train_dataset = DIV2KDataset(root=args.data_path)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)

    print('-' * 20)
    print('Start training')
    print('-' * 20)

    for epoch in range(args.epochs):
        start = timeit.default_timer()
        for _, batch in enumerate(trainloader, 0):
            image, label_hr, label_lr = batch
            for key in optim.keys():
                optim[key].zero_grad()

            '''loss for lr GAN'''

            # D loss for D_1
            image_clean = G_1(image)
            loss_D1 = discriminator_loss(discriminator=D_1, fake=image_clean, real=label_lr)
            loss_D1.backward()
            optim['D_1'].step()

            # GD loss for G_1
            loss_G1 = generator_discriminator_loss(generator=G_1, discriminator=D_1, input=image)
            loss_G1.backward()

            # cycle loss for G_1 and G_2
            loss_cycle = 10 * cycle_loss(G_1, G_2, image)
            loss_cycle.backward()

            # idt loss for G_1
            loss_idt = 5 * identity_loss(clean_image=label_lr, generator=G_1)
            loss_idt.backward()

            # tvloss for G_1
            loss_tv = 0.5 * tvloss(input=image, generator=G_1)
            loss_tv.backward()

            # optimize G_1 and G_2
            optim['G_1'].step()
            optim['G_2'].step()

            print('LR: loss_D1={}, loss_GD={}, loss_cycle={}, loss_idt={}, loss_tv={}'.format(loss_D1.item(),
                                                                                              loss_G1.item(),
                                                                                              loss_cycle.item(),
                                                                                              loss_idt.item(),
                                                                                              loss_tv.item()))

            '''loss for sr GAN'''
            image_clean = G_1(image).detach()
            # D loss for D_2
            image_sr = SR(image_clean)
            loss_D2 = discriminator_loss(discriminator=D_2, fake=image_sr, real=label_hr)
            loss_D2.backward()
            optim['D_2'].step()

            # GD loss for SR
            loss_SR = generator_discriminator_loss(generator=SR, discriminator=D_2, input=image_clean)
            loss_SR.backward()

            # cycle loss for SR and G_3
            loss_cycle = 10 * cycle_loss(SR, G_3, image_clean)
            loss_cycle.backward()

            # idt loss for SR
            loss_idt = 5 * identity_loss_sr(clean_image_lr=label_lr, clean_image_hr=label_hr, generator=SR)
            loss_idt.backward()

            # tvloss for G_1
            loss_tv = 0.5 * tvloss(input=image_clean, generator=SR)
            loss_tv.backward()

            # optimize G_1 and G_2
            optim['SR'].step()
            optim['G_3'].step()

            print('SR: loss_D2={}, loss_SR={}, loss_cycle={}, loss_idt={}, loss_tv={}'.format(loss_D2.item(),
                                                                                              loss_SR.item(),
                                                                                              loss_cycle.item(),
                                                                                              loss_idt.item(),
                                                                                              loss_tv.item()))

        end = timeit.default_timer()
        print('epoch {}, using {} seconds'.format(epoch, end - start))

        torch.save(G_1.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_G_1.pkl'))
        torch.save(G_2.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_G_2.pkl'))
        torch.save(D_1.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_D_1.pkl'))
        torch.save(SR.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_SR.pkl'))
        torch.save(G_3.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_G_3.pkl'))
        torch.save(D_2.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_D_2.pkl'))

    print('Training done.')
    torch.save(G_1.state_dict(), os.path.join(args.log_dir, 'final_weights_G_1.pkl'))
    torch.save(G_2.state_dict(), os.path.join(args.log_dir, 'final_weights_G_2.pkl'))
    torch.save(D_1.state_dict(), os.path.join(args.log_dir, 'final_weights_D_1.pkl'))
    torch.save(SR.state_dict(), os.path.join(args.log_dir, 'final_weights_SR.pkl'))
    torch.save(G_3.state_dict(), os.path.join(args.log_dir, 'final_weights_G_3.pkl'))
    torch.save(D_2.state_dict(), os.path.join(args.log_dir, 'final_weights_D_2.pkl'))

    image = Image.open('/data/data/DIV2K/unsupervused/lr/0001x4d.png')
    image.save('0001x4d.png')
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    sr_image_tensor = SR(G_1(image_tensor))
    sr_image = torchvision.transforms.functional.to_pil_image(sr_image_tensor)
    sr_image.save('0001x4d_sr.png')



if __name__ == '__main__':
    args = parse_args()
    main(args)