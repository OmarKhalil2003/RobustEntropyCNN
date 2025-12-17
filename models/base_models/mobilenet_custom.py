import tensorflow as tf
from tensorflow.keras import layers, models
from models.layers.depthwise_blocks import depthwise_block
from models.layers.attention_se import se_block


def build_mobilenet_custom(
    input_shape=(32, 32, 3),
    num_classes=10,
    use_attention=True
):
    """
    MobileNet-style CNN using depthwise separable convolutions.
    Designed for Phase 1 efficiency + entropy robustness comparison.
    """

    inputs = layers.Input(shape=input_shape)

    # -------------------------------------------------
    # Early convolution (entropy monitoring target)
    # -------------------------------------------------
    x = layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="early_conv"
    )(inputs)
    x = layers.BatchNormalization(name="early_bn")(x)
    x = layers.Activation("relu", name="early_act")(x)

    # -------------------------------------------------
    # Depthwise separable blocks
    # -------------------------------------------------
    x = depthwise_block(x, 64, stride=1, name="mb_block1")
    if use_attention:
        x = se_block(x, name="se1")

    x = depthwise_block(x, 128, stride=2, name="mb_block2")
    if use_attention:
        x = se_block(x, name="se2")

    x = depthwise_block(x, 256, stride=2, name="mb_block3")
    x = depthwise_block(x, 256, stride=1, name="mb_block4")

    # -------------------------------------------------
    # Classification head
    # -------------------------------------------------
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
        inputs,
        outputs,
        name="MobileNetCustom"
    )

    return model
