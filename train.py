import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
from torchvision import transforms as transforms
import numpy as np

import argparse

from dataset import Remote
from unet.unet_model import UNet

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.val_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.val_percent = 0.1
        self.train_loader = None
        self.val_loader = None

    def load_data(self):
        dataset = Remote()
        n_val = int(len(dataset) * self.val_percent)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        self.train_loader = DataLoader(train, batch_size=1, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=1, shuffle=True)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.model = UNet(n_channels=3, n_classes=11).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        self.model.train()
        loss_tol = 0
        n_total = len(self.train_loader) * 512 * 512 * self.train_batch_size
        n_correct = 0
        pbar = tqdm(total=len(self.train_loader))
        for i, (imgs, masks) in enumerate(self.train_loader):
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            imgs = imgs.float()
            output = self.model(imgs)
            loss = self.criterion(output, masks)
            loss.backward()
            self.optimizer.step()
            loss_tol += loss.item()

            pred = torch.max(output, 1).values

            assert pred.shape == masks.shape, 'error'
            n_correct += torch.sum(pred == masks).item()
            pbar.update(imgs.shape[0])
        pbar.close()
        return loss_tol, n_correct/n_total

    def val(self):
        self.model.eval()
        loss_tol = 0
        n_correct = 0
        n_total = len(self.val_loader) * self.val_batch_size * 512 * 512
        with torch.no_grad():
            for i, (img, mask) in enumerate(self.val_loader):
                img, mask = img.to(self.device), mask.to(self.device)
                output = self.model(img)
                loss = self.criterion(output, mask)
                loss_tol += loss.item()

                pred = torch.max(output, 1).values
                n_correct += torch.sum(pred == mask).item()
        return loss_tol, n_correct / n_total

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        try:
            for epoch in range(self.epochs):
                train_result = self.train()
                self.scheduler.step(epoch)

                val_result = self.val()
                print('===> epoch: {0}/{1} val_acc{2}'.format(epoch, self.epochs, val_result[1]))
                accuracy = max(accuracy, val_result[1])
        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), 'INTERRUPTED.pth')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


if __name__ == '__main__':
    main()