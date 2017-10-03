import os.path as path
# import numpy as np
import PIL.Image
import collections
# import torch
from torch.utils import data

class MSCeleb(data.Dataset):
    def __init__(self, root, file_root, transform=None):
        """
        :param root: image folder
        :param file_root: image, label list txt file
        :param transform: do transform on images before training

        """
        self.root = root
        self.file_root = file_root
        self.transform = transform

        self.files = collections.deque()
        imgsets_file = path.join(self.file_root, 'MSCeleb_no_overlap_aligned_origin.txt')
        for line in open(imgsets_file):
            line = line.split(' ')
            img_file = path.join(self.root, line[0])
            lbl = int(line[1])
            self.files.append({'img': img_file, 'lbl': lbl})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]
        img_file = data_file['img']
        img = PIL.Image.open(img_file).convert('RGB')
        lbl = data_file['lbl']
        if self.transform is not None:
            return self.transform(img), lbl
        else:
            return img, lbl