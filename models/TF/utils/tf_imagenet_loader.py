import pathlib
import tensorflow as tf
import glob


path = './dataset/tiny-imagenet-200/'
train_path = path + 'train'
val_path = path + 'val'
test_path = path + 'test'
label_names = sorted(item.split('/')[-1] for item in glob.glob(train_path + '/*'))
label_to_index = dict((name, index) for index, name in enumerate(label_names))


def load_and_preprocess_image(img_path):
    img_raw = tf.io.read_file(img_path)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    # img_tensor = tf.image.resize(img_tensor, [64, 64])
    img_tensor = tf.cast(img_tensor, tf.float16)
    img = img_tensor / 255.0
    return img


def get_images_and_labels(data_root_dir):
    if data_root_dir.split('/')[-1] == 'train':
        img_path = [path for path in sorted(glob.glob(train_path+'/*/images/*.JPEG'))]
        img_label = [label_to_index[img.split('/')[-1].split('_')[0]] for img in img_path]
    
    elif data_root_dir.split('/')[-1] == 'val':
        pass
    
    elif data_root_dir.split('/')[-1] == 'test':
        img_path = [str(path) for path in list(sorted(data_root_dir.glob('**/*.JPEG')))]
        img_label = []
    
    else:
        raise ValueError('Unknown data root dir')
    
    return img_path, img_label
    
def get_dataset(dataset_root_dir):
    img_path, img_label = get_images_and_labels(data_root_dir=dataset_root_dir)
    filename_ds = tf.data.Dataset.from_tensor_slices(img_path)
    image_ds = filename_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(img_label)
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    image_count = len(img_path)
    
    return dataset, image_count

def generate_datasets():
    train_dataset, train_count = get_dataset(train_path)
    val_dataset, val_count = get_dataset(val_path)
    test_dataset, test_count = get_dataset(test_path)
    
    

if __name__ == "__main__":  
    train_dataset, train_count = get_dataset(train_path)
    
    for img, label in train_dataset:
        print(img)