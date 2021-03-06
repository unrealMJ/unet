import argparse

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Remote
from unet.unet_model import UNet


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.train_batch_size
        self.val_batch_size = config.val_batch_size
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
        self.train_loader = DataLoader(train, batch_size=self.train_batch_size, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=self.val_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.model = UNet(n_channels=3, n_classes=11).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-8)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        self.model.train()
        loss_tol = 0
        n_total = len(self.train_loader) * 512 * 512 * self.train_batch_size
        n_correct = 0
        pbar = tqdm(total=len(self.train_loader) * self.train_batch_size, unit='img')
        for i, (imgs, masks) in enumerate(self.train_loader):
            imgs = imgs.to(device=self.device, dtype=torch.float32)
            masks = masks.to(device=self.device, dtype=torch.long)

            output = self.model(imgs)
            loss = self.criterion(output, masks)
            loss_tol += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = torch.max(output, 1)[1]
            correct = torch.sum(pred == masks).item()
            n_correct += correct

            pbar.set_postfix(**{'train_loss': loss.item() / self.train_batch_size,
                                'train_acc': correct / (self.train_batch_size * 512 * 512)})
            pbar.update(imgs.shape[0])
        pbar.close()
        return loss_tol / len(self.train_loader), n_correct / n_total

    def val(self):
        self.model.eval()
        loss_tol = 0
        n_correct = 0
        n_total = len(self.val_loader) * 512 * 512 * self.val_batch_size
        pbar = tqdm(total=len(self.val_loader), desc='val', unit='img')
        with torch.no_grad():
            for i, (imgs, masks) in enumerate(self.val_loader):
                imgs = imgs.to(device=self.device, dtype=torch.float32)
                masks = masks.to(device=self.device, dtype=torch.long)

                output = self.model(imgs)
                loss = self.criterion(output, masks)
                loss_tol += loss.item()

                pred = torch.max(output, 1)[1]
                n_correct += torch.sum(pred == masks).item()

                pbar.update(imgs.shape[0])
        pbar.close()
        return loss_tol / len(self.val_loader), n_correct / n_total

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        writer = SummaryWriter()
        try:
            for epoch in range(self.epochs):
                train_result = self.train()
                # self.scheduler.step(epoch)
                val_result = self.val()
                writer.add_scalar('Loss/train', train_result[0], epoch)
                writer.add_scalar('Acc/train', train_result[1], epoch)
                writer.add_scalar('Loss/val', val_result[0], epoch)
                writer.add_scalar('Acc/val', val_result[1], epoch)
                accuracy = max(accuracy, val_result[1])
                print(f'epoch: {epoch + 1} / {self.epochs} train_loss: {train_result[0]} train_acc: {train_result[1]} '
                      f'val_loss: {val_result[0]} val_acc: {val_result[1]}')
            writer.close()
        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), 'INTERRUPTED.pth')
            writer.close()
        torch.save(self.model.state_dict(), 'model.pth')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=20, type=int, help='number of epochs tp train for')
    parser.add_argument('--train_batch_size', default=2, type=int, help='training batch size')
    parser.add_argument('--val_batch_size', default=1, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


if __name__ == '__main__':
    main()
