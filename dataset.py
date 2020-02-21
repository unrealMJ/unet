import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os


class Remote(Dataset):
    def __init__(self):
        self.img_dir = './data/img/'
        self.mask_dir = './data/mask/'

        imglist = os.listdir(self.img_dir)
        masklist = os.listdir(self.mask_dir)

    def make_path(self, path, item):
        return path + str(item).rjust(4, '0') + '.png'

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, item):
        if item == 29:
            print(23)
        img = cv2.imread(self.make_path(self.img_dir, item))
        mask = cv2.imread(self.make_path(self.mask_dir, item))
        return torch.from_numpy(img), torch.from_numpy(mask)


if __name__ == '__main__':
    dataset = Remote()
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    # for i, (img, mask) in enumerate(train_loader):
    #     print(i, img.shape, mask.shape)
    print(len(train_loader))