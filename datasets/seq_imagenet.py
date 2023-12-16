# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import os
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from datasets.transforms.denormalization import DeNormalize
from augmentations import get_aug
import cv2

class Imagenet(Dataset):
    """
    Defines Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        split = "train" if train else "val"
        imagenet_size = 100
        suffix = ""
        metadata_path = os.path.join(
            root, "{}_{}{}.txt".format(split, imagenet_size, suffix)
        )
        self.data = []
        self.targets = []
        with open(metadata_path) as f:
            for line in f:
                path, target = line.strip().split(" ")

                self.data.append(os.path.join(root, path))
                self.targets.append(int(target))

        self.data = np.array(self.data)
        #self.targets = np.array(self.targets)
        # self.data = np.concatenate(np.array(self.data))
        # self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    @property
    def dims(self):
        return torch.Size([3, 224, 224])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.open(img).convert('RGB')
        #img = cv2.imread(img)
        #print('~~~',img, target)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class SequentialImagenet(ContinualDataset):

    NAME = 'seq-image'
    SETTING = 'task-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20
    TRANSFORM = transforms.Compose(
            [transforms.Resize(224),
             transforms.CenterCrop(84),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def get_data_loaders(self, args):
        transform = get_aug(train=True, **args.aug_kwargs)
        test_transform = get_aug(train=False, train_classifier=False, **args.aug_kwargs)

        train_dataset = Imagenet(base_path() + 'imagenet',
                                 train=True, download=True, transform=transform)

        memory_dataset = Imagenet(base_path() + 'imagenet',
                                 train=True, download=True, transform=test_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
            memory_dataset, _ = get_train_val(memory_dataset, test_transform, self.NAME)
        else:
            test_dataset = Imagenet(base_path() + 'imagenet',
                        train=False, download=True, transform=test_transform)

        train, memory, test = store_masked_loaders(train_dataset, test_dataset, memory_dataset, self)
        return train, memory, test

    def get_transform(self, args):
        imagenet_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        if args.cl_default:
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Resize(224),
                 transforms.CenterCrop(84),
                 transforms.ToTensor(),
                 transforms.Normalize(*imagenet_norm)
                ])
        else:
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Resize(224),
                 transforms.CenterCrop(84),
                 transforms.ToTensor(),
                 transforms.Normalize(*imagenet_norm)
                ])

        return transform

    def not_aug_dataloader(self, batch_size):
        imagenet_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(*imagenet_norm)])

        train_dataset = Imagenet(base_path() + 'imagenet',
                            train=True, download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader
