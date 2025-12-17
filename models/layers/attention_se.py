import tensorflow as tf
from tensorflow.keras import layers

def se_block(x, reduction=16, name=None):
    """
    Squeeze-and-Excitation (SE) block.
    """

    prefix = f"{name}_" if name else ""
    channels = x.shape[-1]

    se = layers.GlobalAveragePooling2D(name=f"{prefix}gap")(x)
    se = layers.Dense(
        channels // reduction,
        activation="relu",
        name=f"{prefix}fc1"
    )(se)
    se = layers.Dense(
        channels,
        activation="sigmoid",
        name=f"{prefix}fc2"
    )(se)

    se = layers.Reshape((1, 1, channels), name=f"{prefix}reshape")(se)
    return layers.Multiply(name=f"{prefix}scale")([x, se])
