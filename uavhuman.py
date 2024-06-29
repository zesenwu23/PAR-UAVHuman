from concurrent.futures import process
from tkinter import Spinbox
import torch
from torch.utils.data import Dataset
import collections
from collections import Counter

import os
from glob import glob
import cv2
import re

from PIL import Image

from augment_and_mix import aug_with_preprocess, augment_and_mix

def get_uavhuman(root, transforms):

    train_split = UAVHuman(os.path.join(root, 'train'), transforms=transforms, split='train', verbose=False)
    test_split = UAVHuman(os.path.join(root, 'test'), transforms=transforms, split='test', verbose=False)

    return train_split, test_split

def imread(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class UAVHuman(Dataset):
    def __init__(self, root, transforms=None, split='train', verbose=True, zero_exit=None):
        super().__init__()
        assert split == 'train' or split == 'test'
        self.root = root
        self.split = split
        self.verbose = verbose

        self.transforms = transforms

        self.pattern = r'P\d+S\d+G(\d+)B(\d+)H(\d+)UC(\d+)LC(\d+)A\d+R\d+_\d+'

        self.fns = sorted(glob(os.path.join(self.root, '*.jpg')))

        if split == 'train':
            self.label2idx = collections.defaultdict(list)
            for item in self.fns:
                g, b, h, ucc, ucs, lcc, lcs = self.parse_label(item)
                self.label2idx['g'].append(int(g)) #[int(g)] = ''
                self.label2idx['b'].append(int(b)) #[int(b)] = ''
                self.label2idx['h'].append(int(h)) #[int(h)] = ''
                self.label2idx['ucc'].append(int(ucc)) #[int(ucc)] = ''
                self.label2idx['ucs'].append(int(ucs)) #[int(ucs)] = ''
                self.label2idx['lcc'].append(int(lcc)) #[int(lcc)] = ''
                self.label2idx['lcs'].append(int(lcs)) #[int(lcs)] = ''
            self.zero_exit = {}
            for key in self.label2idx.keys():
                Counter(self.label2idx[key])
                self.label2idx[key] = set(self.label2idx[key])
                if 0 not in self.label2idx[key]:
                    self.zero_exit[key] = 1
                else:
                    self.zero_exit[key] = 0
        if split == 'test':
            self.zero_exit = zero_exit


    def parse_label(self, fn):
        genders, backpacks, hats, upper_clothes, lower_clothes = re.match(self.pattern, os.path.basename(fn)).groups()
        gender = genders[0]
        backpack = backpacks[0]
        hat = hats[0]
        upper_clothes_color, upper_clothes_style = upper_clothes[0:2], upper_clothes[2:3]
        lower_clothes_color, lower_clothes_style = lower_clothes[0:2], lower_clothes[2:3]

        return gender, backpack, hat, upper_clothes_color, upper_clothes_style, lower_clothes_color, lower_clothes_style
    
    def __getitem__(self, index):
        """
        Labels:
            g   : gender
            b   : backpack
            h   : hat
            ucc : upper_clothes_color
            ucs : upper_clothes_style
            lcc : lower_clothes_color
            lcs : lower_clothes_style
        """
        fn = self.fns[index]
        g, b, h, ucc, ucs, lcc, lcs = self.parse_label(fn)
        
        g = int(g) - self.zero_exit['g']
        b = int(b) - self.zero_exit['b']
        h = int(h) - self.zero_exit['h']
        ucc = int(ucc) - self.zero_exit['ucc']
        ucs = int(ucs) - self.zero_exit['ucs']
        lcc = int(lcc) - self.zero_exit['lcc']
        lcs = int(lcs) - self.zero_exit['lcs']
        
        im = Image.open(fn)
        # im = imread(fn)
        
        if self.transforms is not None:
            im = self.transforms(im)

        return im, torch.Tensor([g, b, h, ucc, ucs, lcc, lcs])

    def __len__(self):
        return len(self.fns)
