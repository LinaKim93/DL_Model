import tensorflow as tf
import tqdm
import math
import numpy as np
import cv2
import albumentations

train_txt_path = '/home/linakim/DL_Model/dataset/tiny-imagenet-200/train.txt'
valid_txt_path = '/home/linakim/DL_Model/dataset/tiny-imagenet-200/train.txt'
img_format = '.JPEG'
num_classes = 200

class Image_Generator(tf.keras.utils.Sequence):
    def __init__(self, dataset_info_path, batch_size, input_shape, num_classes, shuffle):
        self.dataset_info_path = dataset_info_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.data = self.get_dataset(dataset_info_path)
        self.on_epoch_end()
    
    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)
            
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        data = [self.data[i] for i in indexes]
        x, y = self.__data_generation(data)
        return x, y
    
    def __data_generation(self, data):
        batch_img = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        batch_cls = np.zeros(shape=(self.batch_size, self.num_classes), dtype=np.float32)
        img_list = []
        cls_list = []
        for img_path in data:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
            img_list.append(img)
            txt_path = img_path.replace(img_format, '.txt')
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if idx == 0:
                        cls = line.replace('\n', '')
                        cls_list.append(cls)
                        break                    
        
        for i in range(len(data)):
            cls = cls_list[i]
            cls = tf.keras.utils.to_categorical(cls, num_classes=self.num_classes)
            batch_img[i] = img_list[i] / 255.0
            batch_cls[i] = cls
        
        return batch_img, batch_cls    
                
    
    def get_dataset(self, dataset_info_path):
        img_list = []
        with open(dataset_info_path, 'r') as f:
            lines = f.readlines()
            for img_path in lines:
                img_path = img_path.replace('\n', '')
                img_list.append(img_path)
        return img_list
                
                
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    
if __name__ == "__main__":
    gen = Image_Generator(dataset_info_path=train_txt_path, batch_size=16, input_shape=(64, 64, 3), num_classes=num_classes, shuffle=True)
    for i in tqdm.tqdm(range(gen.__len__())):
        gen.__getitem__(i)