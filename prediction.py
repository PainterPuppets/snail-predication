import keras
from dataset import load_dataset, AUTOTUNE
import matplotlib.pyplot as plt
import numpy  as np
import tensorflow as tf
tf.random.set_seed(1)


BATCH_SIZE = 6
new_model = keras.models.load_model('./model/snail_predicate.keras')
dataset, train_image_label, train_image_paths = load_dataset()

train_ds = dataset.take(400).shuffle(1000)
val_ds   = dataset.skip(200).shuffle(1000)

train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

plt.figure(figsize=(10, 5))  # 图形的宽为10高为5

for images, labels in val_ds.take(1):
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)  
        
        # 显示图片
        plt.imshow(images[i])
        
        # 需要给图片增加一个维度
        img_array = tf.expand_dims(images[i], 0) 
        
        # 使用模型预测路标
        predictions = new_model.predict(img_array)
        plt.title(np.argmax(predictions))

        plt.axis("off")