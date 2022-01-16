import models.LeNet.lenet as lenet
import imageio
import numpy as np
import cv2


def set_dict():
    dict_list = {}
    with open('dataset/words.txt', 'r') as f:
        for line in f.readlines():
            code = line.split('\t')[0].strip()
            obj = line.split('\t')[1].strip()
            dict_list[code] = obj

    return dict_list


def train():
    img = imageio.imread('dataset/train/n01443537/images/n01443537_1.JPEG')
    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    dict_list = set_dict()
    # model = lenet.LeNet(input_shape=(28, 28, 3))
    # model.summary()
    train()
