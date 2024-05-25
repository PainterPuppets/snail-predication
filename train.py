import matplotlib.pyplot as plt
import tensorflow as tf
from model import InceptionResNetV2
from dataset import load_dataset, AUTOTUNE

tf.random.set_seed(1)
BATCH_SIZE = 4
epochs = 15

def train(train_ds, val_ds):
    model = InceptionResNetV2([299,299,3],58)
    model.summary()

    initial_learning_rate = 1e-4

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, 
            decay_steps=100,
            decay_rate=0.96,
            staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return model, history

def draw_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)

    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

dataset, train_image_label, train_image_paths = load_dataset()

train_ds = dataset.take(400).shuffle(1000)
val_ds   = dataset.skip(200).shuffle(1000)

train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

model, history = train(train_ds, val_ds)
draw_history(history)

# 保存模型
model.save('model/snail_predicate.keras')


