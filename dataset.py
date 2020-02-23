import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Remote(Dataset):
    def __init__(self):
        self.img_dir = './data/img/'
        self.mask_dir = './data/mask/'

    def make_path(self, path, item):
        return path + str(item).rjust(4, '0') + '.png'

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, item):
        img = cv2.imread(self.make_path(self.img_dir, item)).transpose((2, 0, 1))
        mask = cv2.imread(self.make_path(self.mask_dir, item), cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(img), torch.from_numpy(mask)


if __name__ == '__main__':
    dataset = Remote()
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    img0_true = cv2.imread('./data/img/0000.png').transpose((2, 0, 1))
    img0_true = torch.from_numpy(img0_true)
    print(len(train_loader))

    for img, mask in train_loader:
        img = img[0]
        print(img.shape, mask.shape)
        assert torch.sum(img == img0_true) / (512 * 512 * 3) == 1