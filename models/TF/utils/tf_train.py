from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__)))))))))

import tensorflow as tf
import numpy as np
from models.TF.utils.tf_imagenet_loader import *
from models.TF.Classifier.LeNet.Lenet import LeNet


def train():
    batch_size = 8
    classes = 200
    epoches = 10
    
    img_width, img_height = (64, 64)
    img_channels = 3
    
    train_data, train_labels, val_data, val_labels = get_data_classification()
    # train_data = 

    input_shape = (img_width, img_height, img_channels)
    input_tensor = tf.keras.layers.Input(input_shape)
    model = LeNet(input_tensor)
    model.summary()
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # log_dir = 'logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # model.fit(train_data, train_labels, epochs=epoches, batch_size=batch_size, validation_data=(val_data, val_labels), callbacks=[tensorboard_callback])
    model.fit(train_data, train_labels, epochs=epoches, batch_size=batch_size, validation_data=(val_data, val_labels))    


if __name__ == "__main__":
    train()
