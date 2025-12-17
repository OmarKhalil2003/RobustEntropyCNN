import tensorflow as tf
from tensorflow.keras import layers

def grouped_conv_block(
    x,
    filters,
    kernel_size=3,
    groups=2,
    stride=1,
    activation="relu",
    name=None
):
    """
    Grouped convolution block (Conv2D with groups).
    """

    prefix = f"{name}_" if name else ""

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        groups=groups,
        use_bias=False,
        name=f"{prefix}gconv"
    )(x)
    x = layers.BatchNormalization(name=f"{prefix}bn")(x)
    x = layers.Activation(activation, name=f"{prefix}act")(x)

    return x
