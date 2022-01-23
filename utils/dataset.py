import glob
import os
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


def CreateTrainDataSet(path):
    """
    This function returns a list of all files in a directory
    """
    # Get all files in the directory
    train_path = os.path.join(path, 'train/*/*/*.JPEG')
    train_img_path = glob.glob(train_path)
    train_img = np.array([cv2.imread(img) for img in train_img_path])
    train_label = np.array(
        [img.split('/')[-1].split('.')[0].split('_')[0] for img in train_img_path])

    return train_img, train_label


def GetValData(path):
    pass
