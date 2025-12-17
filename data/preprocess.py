import tensorflow as tf

def get_preprocessing_layer(augment: bool = True):
    """
    Returns a Keras preprocessing layer.
    Adversarial-safe: keeps values in [0, 1].
    """

    if not augment:
        return tf.keras.layers.Lambda(lambda x: x)

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ],
        name="data_augmentation"
    )
