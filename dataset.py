import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)  # 编码解码处理
    image = tf.image.resize(image, [299,299])        # 图片调整
    return image/255.0                               # 归一化处理

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def load_dataset():
    t_pictures_dir = pathlib.Path("./pictures/T")
    f_pictures_dir = pathlib.Path("./pictures/F")

    trains = []

    for img in t_pictures_dir.glob('*.png'):
        trains.append([str(img.absolute()), 1])
    
    for img in f_pictures_dir.glob('*.png'):
        trains.append([str(img.absolute()), 0])

    print('trains len', len(trains))

    random.shuffle(trains)

    train_image_label = [i[1] for i in trains]
    train_label_ds = tf.data.Dataset.from_tensor_slices(train_image_label)

    train_image_paths = [i[0] for i in trains]
    train_path_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)

    train_image_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    image_label_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))

    return image_label_ds, train_image_label, train_image_paths
  

def show_dataset(train_image_paths, train_image_label):
    plt.figure(figsize=(20,4))

    for i in range(20):
        plt.subplot(2,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        
        # 显示图片
        images = plt.imread(train_image_paths[i])
        plt.imshow(images)
        # 显示标签
        plt.xlabel(train_image_label[i])

    plt.show()
