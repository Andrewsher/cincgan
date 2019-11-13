import os
import torch
from torch.utils import data
import torchvision
from PIL import Image

# import imgaug.augmenters as iaa
# from imgaug.augmentables.segmaps import SegmentationMapOnImage



class DIV2KDataset(data.Dataset):
    def __init__(self, root, crop_size=16, is_training=True):
        self.root = root
        self.crop_size = crop_size
        self.is_training = is_training
        self.hr_file_names = os.listdir(os.path.join(self.root, 'hr'))
        self.lr_file_names = os.listdir(os.path.join(self.root, 'lr'))
        # self.seq = iaa.Sequential([
        #     iaa.Fliplr(0.5),
        #     iaa.Flipud(0.5),
        #     iaa.Rot90(k=[0, 1, 2, 3])
        # ])

    def __len__(self):
        return len(self.lr_file_names)

    def __getitem__(self, item):
        lr_file_name = self.lr_file_names[item % len(self.lr_file_names)]
        image = Image.open(os.path.join(self.root, 'lr', lr_file_name))
        # image = self.seq(images=image)
        image = torchvision.transforms.ToTensor()(image)

        hr_file_name = self.hr_file_names[item % len(self.hr_file_names)]
        hr_label = Image.open(os.path.join(self.root, 'hr', hr_file_name))
        hr_label = torchvision.transforms.ToTensor()(hr_label)
        lr_label = hr_label.resize((hr_label.shape[0]//4, hr_label.shape[1]//4), Image.BICUBIC)
        lr_label = torchvision.transforms.ToTensor()(lr_label)

        # det = self.seq.to_deterministic()
        # batch_images = det.augment_image(image)
        # batch_labels = det.augment_segmentation_maps(batch_labels)
        # batch_images, batch_labels = self.seq(image=batch_images, segmentation_maps=batch_labels)

        return image, hr_label, lr_label