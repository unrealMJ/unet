import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
import time


class Remote(Dataset):
    def __init__(self):
        self.img_dir = './data/img/'
        self.mask_dir = './data/mask/'

    def make_path(self, path, item):
        return path + str(item).rjust(4, '0') + '.png'

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, item):
        img = cv2.imread(self.make_path(self.img_dir, item)).reshape(3, 512, 512)
        mask = cv2.imread(self.make_path(self.mask_dir, item)).reshape(3, 512, 512)
        return torch.from_numpy(img), torch.from_numpy(mask)


if __name__ == '__main__':
    dataset = Remote()
    train_loader = DataLoader(dataset=dataset, batch_size=1)
    print(len(train_loader))
    for i, (img, mask) in enumerate(train_loader):
        print(i, img.shape, mask.shape)
