import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__)))))))))

import numpy as np
import cv2
import imageio
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.TF.utils.tf_imagenet_loader import *


def train():
    batch_size = 64
    classes = 200
    epoches = 10
    
    img_width, img_height = (64, 64)
    img_channels = 3
    
    train_data, train_labels, val_data, val_labels = get_data_classification(get_id_dict())
        
    # normalize
    


if __name__ == "__main__":
    train()
