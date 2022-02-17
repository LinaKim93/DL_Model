import time
import cv2
import numpy as np


path = 'dataset/tiny-imagenet-200/'


def get_id_dict():
    id_dict = {}
    for i, line in enumerate(open(path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict


def get_class_to_id_dict():
    id_dict = get_id_dict()
    all_classes = {}
    result = {}
    for i, line in enumerate(open(path + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])      
    return result


def get_data_classification(id_dict):
    print("loading data")
    train_data, val_data = [], []
    train_labels, val_labels = [], []
    t = time.time()
    for key, value in id_dict.items():
        train_data += [cv2.imread(path+'train/{}/images/{}_{}.JPEG'.format(key, key, str(i))) for i in range(500)]
        train_labels_ = np.array([[0]*200]*500)
        train_labels_[:, value] = 1
        train_labels += train_labels_.tolist()
        
    for line in open(path+'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        val_data.append(cv2.imread(path+'val/images/{}'.format(img_name)))
        val_labels_ = np.array([[0]*200])
        val_labels_[0, id_dict[class_id]] = 1
        val_labels += val_labels_.tolist()
        
    print('finished loading data, in {} seconds'.format(time.time()-t))
    return np.array(train_data), np.array(train_labels), np.array(val_data), np.array(val_labels)


def get_data_bboxes():
    # BBOX annotation data
    pass

if __name__ == "__main__":
    # batch_size = 64
    
    train_data, train_labels, val_data, val_labels = get_data_classification(get_id_dict())
    print( "train data shape: ",  train_data.shape )
    print( "train label shape: ", train_labels.shape )
    print( "test data shape: ",   val_data.shape )
    print( "test_labels.shape: ", val_labels.shape )