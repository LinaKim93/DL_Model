import time
import cv2
import numpy as np
import glob


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
        result[key] = all_classes[key]
    return result


def get_data_classification():
    print("loading data")
    t = time.time()
    train_data = sorted(glob.glob(path+'train/*/images/*.JPEG'))
    
    X_train = np.zeros((100000 , 64, 64, 3), dtype=np.float32)
    Y_train = []
    X_val = np.zeros((10000, 64, 64, 3), dtype=np.float32)
    Y_val = []

    # X_train = [imageio.imread(img, as_gray=False, pilmode='RGB') for img in train_data]
    # Y_train = [img.split('/')[-3] for img in train_data]
    for i in range(len(train_data)):
        X_train[i] = cv2.imread(train_data[i], cv2.IMREAD_COLOR)
        Y_train.append(train_data[i].split('/')[-3])
    
    i = 0
    for line in open(path+'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        X_val[i] = cv2.imread(path+'val/images/'+img_name, cv2.IMREAD_COLOR)
        Y_val.append(class_id)
        i += 1
        
    print('finished loading data, in {} seconds'.format(time.time()-t))
    
    return X_train, np.array(Y_train), X_val, np.array(Y_val)


def get_data_bboxes():
    # BBOX annotation data
    pass


def shuffle_data(train_data, train_labels):
    size = len(train_data)
    train_idx = np.arange(size)
    np.random.shuffle(train_idx)
    return train_data[train_idx], train_labels[train_idx]


if __name__ == "__main__":  
    train_data, train_labels, val_data, val_labels = get_data_classification()
    print( "train data shape: ",  train_data.shape )
    print( "train label shape: ", train_labels.shape )
    print( "test data shape: ",   val_data.shape )
    print( "test_labels shape: ", val_labels.shape )
    
    cls_dict = get_class_to_id_dict()
    train_data /= 255.0
    val_data /= 255.0
    
    print(train_data[0])
    print(val_labels[0], cls_dict[val_labels[0]])