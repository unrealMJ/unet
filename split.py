import cv2
import numpy as np
import os

di = {
    0: [0, 229, 254],
    1: [0, 254, 162],
    2: [254, 212, 0],
    3: [0, 5, 255],
    4: [255, 101, 0],
    5: [0, 255, 50],
    6: [0, 0, 0],
    7: [0, 117, 255],
    8: [178, 254, 0],
    9: [67, 255, 0],
    10: [255, 0, 16]
}


def make_path(item):
    item = str(item).rjust(4, '0')
    return './data/img/{0}.png'.format(item)


def process(mask):
    new_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            new_mask[row][col] = get_key(mask[row][col])
        if row % 10 == 0:
            print(row)
    return new_mask


def get_key(value):
    for key in di:
        if (di[key] == value).all():
            return key
    return None


index = 0


def split(img, begin_x=0, begin_y=0, h=512, w=512, step=50):
    global index
    while begin_y + h < img.shape[0]:
        begin_x = 0
        while begin_x + w < img.shape[1]:
            patch = img[begin_y:begin_y + h, begin_x:begin_x + w]
            begin_x += step
            path = make_path(index)
            cv2.imwrite(path, patch)
            index += 1
        begin_y += step


if __name__ == '__main__':
    path = './raw/img/'
    for each in os.listdir(path):
        img = cv2.imread(os.path.join(path, each))
        split(img)

    # path = './raw/mask/'
    # for each in os.listdir(path):
    #     img = cv2.imread(os.path.join(path, each))
    #     split(process(img))