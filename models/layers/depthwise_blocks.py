import tensorflow as tf
from tensorflow.keras import layers

def depthwise_block(
    x,
    pointwise_filters,
    stride=1,
    activation="relu",
    name=None
):
    """
    Depthwise separable convolution block:
    DepthwiseConv → BN → Act → PointwiseConv → BN → Act
    """

    prefix = f"{name}_" if name else ""

    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{prefix}dwconv"
    )(x)
    x = layers.BatchNormalization(name=f"{prefix}dw_bn")(x)
    x = layers.Activation(activation, name=f"{prefix}dw_act")(x)

    x = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=f"{prefix}pwconv"
    )(x)
    x = layers.BatchNormalization(name=f"{prefix}pw_bn")(x)
    x = layers.Activation(activation, name=f"{prefix}pw_act")(x)

    return x
