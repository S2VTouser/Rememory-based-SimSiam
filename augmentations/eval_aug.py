from torchvision import transforms
from PIL import Image
import torch

imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
#imagenet_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]


class Transform_single():
    def __init__(self, image_size, train, normalize=imagenet_norm):
        if train == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                                             interpolation=Image.BICUBIC),
                # transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])
            self.transform1 = transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),   #(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])
            self.transform1 = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform1(x)
        return x1   #, x2
