import tensorflow as tf
from tensorflow.keras import layers, models
from models.layers.grouped_conv import grouped_conv_block


def residual_grouped_block(
    x,
    filters,
    groups=4,
    stride=1,
    name=None
):
    """
    Residual block with grouped convolutions.
    """
    shortcut = x
    prefix = f"{name}_" if name else ""

    # First grouped conv
    x = grouped_conv_block(
        x,
        filters=filters,
        kernel_size=3,
        groups=groups,
        stride=stride,
        name=f"{prefix}conv1"
    )

    # Second grouped conv
    x = grouped_conv_block(
        x,
        filters=filters,
        kernel_size=3,
        groups=groups,
        stride=1,
        name=f"{prefix}conv2"
    )

    # Projection shortcut if shape mismatch
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"{prefix}proj"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{prefix}proj_bn")(shortcut)

    # Residual add
    x = layers.Add(name=f"{prefix}add")([x, shortcut])
    x = layers.Activation("relu", name=f"{prefix}out")(x)

    return x


def build_resnet_custom(
    input_shape=(32, 32, 3),
    num_classes=10,
    groups=4
):
    """
    Grouped-ResNet CNN for RobustEntropyCNN (Phase 1).
    """

    inputs = layers.Input(shape=input_shape)

    # --------------------------------------------------
    # Early convolution (ENTROPY MONITOR TARGET)
    # --------------------------------------------------
    x = layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="early_conv"
    )(inputs)
    x = layers.BatchNormalization(name="early_bn")(x)
    x = layers.Activation("relu", name="early_act")(x)

    # --------------------------------------------------
    # Residual stages (grouped convs)
    # --------------------------------------------------
    x = residual_grouped_block(
        x, filters=64, groups=groups, stride=1, name="res_block1"
    )

    x = residual_grouped_block(
        x, filters=128, groups=groups, stride=2, name="res_block2"
    )

    x = residual_grouped_block(
        x, filters=256, groups=groups, stride=2, name="res_block3"
    )

    # --------------------------------------------------
    # Classification head
    # --------------------------------------------------
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    logits = layers.Dense(
        num_classes,
        name="logits"
    )(x)

    outputs = layers.Activation(
        "softmax",
        name="softmax"
    )(logits)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNetGrouped"
    )

    return model
