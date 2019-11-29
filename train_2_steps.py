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
from torch.utils.tensorboard import SummaryWriter
from time import sleep

from data import DIV2KDataset
from models import Discriminator_sr, Discriminator_lr, EDSR, Generator_sr, Generator_lr
from loss import generator_discriminator_loss, discriminator_loss, cycle_loss, identity_loss, identity_loss_sr, tvloss
from test import resolv_sr, resolv_deonoise


def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # config
    parser.add_argument('-d', '--data-path', default='/data/data/DIV2K/unsupervised/train_dataset', type=str)
    parser.add_argument('-g', '--gpu', default=0, type=int)
    parser.add_argument('-l', '--log-dir', default='output-0/', type=str)
    parser.add_argument('-c', '--in-channels', default=3, type=int)
    parser.add_argument('-w', '--in-w', default=16, type=int)
    # Train Setting
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
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

    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    return args


def main(args):
    os.makedirs(args.log_dir, exist_ok=True)
    iter_index = 0

    # tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    # get dataloader
    train_dataset = DIV2KDataset(root=args.data_path)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size * 2, shuffle=True, num_workers=3)

    '''step 1: train LR model'''
    # create models
    G_1 = Generator_lr(in_channels=args.in_channels)
    G_2 = Generator_lr(in_channels=args.in_channels)
    D_1 = Discriminator_lr(in_channels=args.in_channels, in_h=args.in_w, in_w=args.in_w)

    for model in [G_1, G_2, D_1]:
        model.cuda()
        model.train()

    # create optimizors
    optim = {
        'G_1': torch.optim.Adam(params=filter(lambda p: p.requires_grad, G_1.parameters()), lr=args.lr),
        'G_2': torch.optim.Adam(params=filter(lambda p: p.requires_grad, G_2.parameters()), lr=args.lr),
        'D_1': torch.optim.Adam(params=filter(lambda p: p.requires_grad, D_1.parameters()), lr=args.lr)
    }
    for key in optim.keys():
        optim[key].zero_grad()

    print('-' * 20)
    print('Start training LR')
    print('-' * 20)
    for epoch in range(0, args.epochs // 4):
        G_1.train()
        D_1.train()
        G_2.train()
        start = timeit.default_timer()
        for _, batch in enumerate(trainloader):
            iter_index += 1
            image, _, label_lr = batch

            image = image.cuda()
            label_lr = label_lr.cuda()

            '''loss for lr GAN'''
            '''update G_1 and G_2'''
            optim['D_1'].zero_grad()
            optim['G_1'].zero_grad()
            optim['G_2'].zero_grad()
            # D loss for D_1
            # if iter_index % 10 == 0:
            image_clean_d = G_1(image).detach()
            loss_D1 = discriminator_loss(discriminator=D_1, fake=image_clean_d, real=label_lr) / 1000.
            loss_D1.backward()
            optim['D_1'].step()

            # GD loss for G_1
            loss_G1 = generator_discriminator_loss(generator=G_1, discriminator=D_1, input=image)
            # loss_G1.backward()

            # cycle loss for G_1 and G_2
            loss_cycle = 10 * cycle_loss(G_1, G_2, image)
            # loss_cycle.backward()

            # idt loss for G_1
            loss_idt = 5 * identity_loss(clean_image=label_lr, generator=G_1)
            # loss_idt.backward()

            # tvloss for G_1
            loss_tv = 0.5 * tvloss(input=image, generator=G_1)
            # loss_tv.backward()

            # loss functions
            loss = loss_G1 + loss_cycle + loss_idt + loss_tv
            loss.backward()

            # optimize D_1, G_1 and G_2
            optim['D_1'].step()
            optim['G_1'].step()
            optim['G_2'].step()

            if iter_index % 100 == 0:
                print('iter {}: LR: loss_GD={}, loss_cycle={}, loss_idt={}, loss_tv = {}'.format(iter_index,
                                                                                                             # loss_D1.item(),
                                                                                                             loss_G1.item(),
                                                                                                             loss_cycle.item(),
                                                                                                             loss_idt.item(),
                                                                                                             loss_tv.item()))
                writer.add_scalar('LR/loss_D1', loss_D1.item(), iter_index // 100)
                writer.add_scalar('LR/loss_GD', loss_G1.item(), iter_index // 100)
                writer.add_scalar('LR/loss_cycle', loss_cycle.item(), iter_index // 100)
                writer.add_scalar('LR/loss_idt', loss_idt.item(), iter_index // 100)
                writer.add_scalar('LR/loss_tv', loss_tv.item(), iter_index // 100)
                writer.add_image('LR/origin', image[0], iter_index // 100)
                writer.add_image('LR/denoise', G_1(image)[0], iter_index // 100)
                writer.flush()

        end = timeit.default_timer()
        print('epoch {}, using {} seconds'.format(epoch, end - start))

        G_1.eval()
        image = Image.open('/data/data/DIV2K/unsupervised/lr/0001x4d.png')
        clean_image = resolv_deonoise(G_1, image)
        # image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0).cuda()
        # sr_image_tensor = SR(G_1(image_tensor).detach())
        # sr_image = torchvision.transforms.functional.to_pil_image(sr_image_tensor[0].cpu())
        clean_image.save(os.path.join(args.log_dir, '0001x4d_clean_{}.png'.format(str(epoch))))

        torch.save(G_1.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_G_1.pkl'))
        torch.save(G_2.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_G_2.pkl'))
        torch.save(D_1.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_D_1.pkl'))

    print('Training denoising model done.')
    torch.save(G_1.state_dict(), os.path.join(args.log_dir, 'weights_step_1_G_1.pkl'))
    torch.save(G_2.state_dict(), os.path.join(args.log_dir, 'weights_step_1_G_2.pkl'))
    torch.save(D_1.state_dict(), os.path.join(args.log_dir, 'weights_step_1_D_1.pkl'))

    image = Image.open('/data/data/DIV2K/unsupervised/lr/0001x4d.png')
    image.save(os.path.join(args.log_dir, '0001x4d.png'))
    clean_image = resolv_deonoise(G_1, image)
    clean_image.save(os.path.join(args.log_dir, '0001x4d_clean.png'))

    # ''' clean cache'''
    # del G_1, G_2, D_1, optim
    # torch.cuda.empty_cache()
    # sleep(5)

    '''step 2: train SR model'''
    # create models
    SR = EDSR(n_colors=args.in_channels)
    G_3 = Generator_sr(in_channels=args.in_channels)
    D_2 = Discriminator_sr(in_channels=args.in_channels, in_h=args.in_w * 4, in_w=args.in_w * 4)

    # load pretrained model
    # G_1.load_state_dict(torch.load(os.path.join(args.log_dir, 'weights_step_1_G_1.pkl')))

    # get dataloader
    del trainloader, train_dataset
    train_dataset = DIV2KDataset(root=args.data_path)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)

    for model in [G_1, G_2, D_1, SR, G_3, D_2]:
        model.cuda()
        model.train()

    # create optimizors
    del optim
    optim = {
        'G_1': torch.optim.Adam(params=filter(lambda p: p.requires_grad, G_1.parameters()), lr=args.lr),
        'G_2': torch.optim.Adam(params=filter(lambda p: p.requires_grad, G_2.parameters()), lr=args.lr),
        'D_1': torch.optim.Adam(params=filter(lambda p: p.requires_grad, D_1.parameters()), lr=args.lr),
        'SR': torch.optim.Adam(params=filter(lambda p: p.requires_grad, SR.parameters()), lr=args.lr),
        'G_3': torch.optim.Adam(params=filter(lambda p: p.requires_grad, G_3.parameters()), lr=args.lr),
        'D_2': torch.optim.SGD(params=filter(lambda p: p.requires_grad, D_2.parameters()), lr=args.lr)
    }
    for key in optim.keys():
        optim[key].zero_grad()

    print('-' * 20)
    print('Start training SR')
    print('-' * 20)
    for epoch in range(args.epochs // 4, args.epochs):
        G_1.train()
        SR.train()
        start = timeit.default_timer()
        for _, batch in enumerate(trainloader):
            iter_index += 1
            image, label_hr, label_lr = batch
            image = image.cuda()
            label_hr = label_hr.cuda()
            label_lr = label_lr.cuda()

            '''loss for lr GAN'''
            '''update G_1 and G_2'''
            optim['D_1'].zero_grad()
            optim['G_1'].zero_grad()
            optim['G_2'].zero_grad()

            # D loss for D_1
            image_clean_d = G_1(image).detach()
            loss_D1 = discriminator_loss(discriminator=D_1, fake=image_clean_d, real=label_lr) / 1000.
            loss_D1.backward()
            optim['D_1'].step()

            # GD loss for G_1
            loss_G1 = generator_discriminator_loss(generator=G_1, discriminator=D_1, input=image)
            # loss_G1.backward()

            # cycle loss for G_1 and G_2
            loss_cycle = 10 * cycle_loss(G_1, G_2, image)
            # loss_cycle.backward()

            # idt loss for G_1
            loss_idt = 1 * identity_loss(clean_image=label_lr, generator=G_1)
            # loss_idt.backward()

            # tvloss for G_1
            loss_tv = 0.5 * tvloss(input=image, generator=G_1)
            # loss_tv.backward()

            # loss functions
            loss = loss_G1 + loss_cycle + loss_idt + loss_tv
            loss.backward()

            # optimize D_1, G_1 and G_2
            optim['D_1'].step()
            optim['G_1'].step()
            optim['G_2'].step()

            if iter_index % 100 == 0:
                print('iter {}: LR: loss_GD={}, loss_cycle={}, loss_idt={}, loss_tv = {}'.format(iter_index,
                                                                                                 # loss_D1.item(),
                                                                                                 loss_G1.item(),
                                                                                                 loss_cycle.item(),
                                                                                                 loss_idt.item(),
                                                                                                 loss_tv.item()))
                writer.add_scalar('LR/loss_D1', loss_D1.item(), iter_index // 100)
                writer.add_scalar('LR/loss_GD', loss_G1.item(), iter_index // 100)
                writer.add_scalar('LR/loss_cycle', loss_cycle.item(), iter_index // 100)
                writer.add_scalar('LR/loss_idt', loss_idt.item(), iter_index // 100)
                writer.add_scalar('LR/loss_tv', loss_tv.item(), iter_index // 100)
                writer.add_image('LR/origin', image[0], iter_index // 100)
                writer.add_image('LR/denoise', G_1(image)[0], iter_index // 100)
                writer.flush()

            '''loss for sr GAN'''
            '''update G_1, SR and G_3'''
            for key in optim.keys():
                optim[key].zero_grad()

            image_clean = G_1(image)
            image_clean_detach = image_clean.detach()
            # D loss for D_2
            image_sr = SR(image_clean_detach)
            loss_D2 = discriminator_loss(discriminator=D_2, fake=image_sr, real=label_hr) / 1000.
            loss_D2.backward()
            optim['D_2'].step()

            # GD loss for SR and G_1
            loss_SR = generator_discriminator_loss(generator=SR, discriminator=D_2, input=image_clean)
            # loss_SR.backward()

            # cycle loss for SR and G_3
            loss_cycle = 10 * cycle_loss(SR, G_3, image_clean_detach)
            # loss_cycle.backward()

            # idt loss for SR
            loss_idt = 5 * identity_loss_sr(clean_image_lr=label_lr, clean_image_hr=label_hr, generator=SR)
            # loss_idt.backward()

            # tvloss for SR
            loss_tv = 2 * tvloss(input=image_clean, generator=SR)
            # loss_tv.backward()

            loss = loss_SR +loss_cycle + loss_idt + loss_tv
            loss.backward()

            # optimize G_1, SR and G_3
            optim['D_2'].step()
            optim['G_1'].step()
            optim['SR'].step()
            optim['G_3'].step()

            if iter_index % 100 == 0:
                print(
                    'iter {}: SR: loss_SR={}, loss_cycle={}, loss_idt={}, loss_tv={}'.format(iter_index,
                                                                                             # loss_D2.item(),
                                                                                             loss_SR.item(),
                                                                                             loss_cycle.item(),
                                                                                             loss_idt.item(),
                                                                                             loss_tv.item()))
                writer.add_scalar('SR/loss_D2', loss_D2.item(), iter_index // 100)
                writer.add_scalar('SR/loss_G1', loss_G1.item(), iter_index // 100)
                writer.add_scalar('SR/loss_SR', loss_SR.item(), iter_index // 100)
                writer.add_scalar('SR/loss_cycle', loss_cycle.item(), iter_index // 100)
                writer.add_scalar('SR/loss_idt', loss_idt.item(), iter_index // 100)
                writer.add_scalar('SR/loss_tv', loss_tv.item(), iter_index // 100)
                writer.add_image('SR/origin', image[0], iter_index // 100)
                writer.add_image('SR/clean_image', G_1(image)[0], iter_index // 100)
                writer.add_image('SR/SR', SR(G_1(image))[0], iter_index // 100)
                writer.flush()

        end = timeit.default_timer()
        print('epoch {}, using {} seconds'.format(epoch, end - start))

        G_1.eval()
        SR.eval()
        image = Image.open('/data/data/DIV2K/unsupervised/lr/0001x4d.png')
        sr_image = resolv_sr(G_1, SR, image)
        sr_image.save(os.path.join(args.log_dir, '0001x4d_sr_{}.png'.format(str(epoch))))

        torch.save(G_1.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_G_1.pkl'))
        torch.save(G_2.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_G_2.pkl'))
        torch.save(D_1.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_D_1.pkl'))
        torch.save(SR.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_SR.pkl'))
        torch.save(G_3.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_G_3.pkl'))
        torch.save(D_2.state_dict(), os.path.join(args.log_dir, 'ep-' + str(epoch) + '_D_2.pkl'))

    writer.close()
    print('Training done.')
    torch.save(G_1.state_dict(), os.path.join(args.log_dir, 'final_weights_G_1.pkl'))
    torch.save(SR.state_dict(), os.path.join(args.log_dir, 'final_weights_SR.pkl'))
    torch.save(G_3.state_dict(), os.path.join(args.log_dir, 'final_weights_G_3.pkl'))
    torch.save(D_2.state_dict(), os.path.join(args.log_dir, 'final_weights_D_2.pkl'))

    image = Image.open('/data/data/DIV2K/unsupervised/lr/0001x4d.png')
    image.save(os.path.join(args.log_dir, '0001x4d.png'))
    sr_image = resolv_sr(G_1, SR, image)
    sr_image.save(os.path.join(args.log_dir, '0001x4d_sr.png'))


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main(args)
