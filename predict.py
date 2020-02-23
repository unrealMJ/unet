import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from split import di
from unet.unet_model import UNet


class Prediction(object):
    def __init__(self):
        self.model = None
        self.criterion = None
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

    def load_model(self):
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load('model.pth', map_location=self.device))
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def predict(self, img):
        img = img.transpose((2, 0, 1))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = torch.from_numpy(img)
        pred = self.model(img)
        return pred

    def visualize(self, pred):
        pred = torch.max(pred, 1)[1].numpy()
        mask = np.zeros_like(pred)
        for row in range(pred.shape[0]):
            for col in range(pred.shape[1]):
                mask[row][col] = di[pred[row][col]]
        return mask

    def eval(self, mask, ground_truth):
        loss = self.criterion(mask, ground_truth)
        return loss


if __name__ == '__main__':
    img = cv2.imread('./test/5.tif')
    ground_truth = cv2.imread('./test/5-m.tif')
    obj = Prediction()
    obj.load_model()
    mask = obj.visualize(obj.predict(img))
    cv2.imwrite('./mask.png', mask)

