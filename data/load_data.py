import tensorflow as tf
from typing import Tuple

def load_cifar10(
    batch_size: int = 128,
    val_split: float = 0.1,
    shuffle: bool = True
):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    y_train = tf.squeeze(y_train)
    y_test = tf.squeeze(y_test)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    val_size = int(len(x_train) * val_split)

    x_val = x_train[:val_size]
    y_val = y_train[:val_size]

    x_train = x_train[val_size:]
    y_train = y_train[val_size:]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=10000)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds
