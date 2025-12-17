import tensorflow as tf
from tensorflow.keras import layers, models
from models.layers.depthwise_blocks import depthwise_block
from models.layers.attention_se import se_block

def build_simple_cnn(
    input_shape=(32, 32, 3),
    num_classes=10,
    use_attention=True
):
    inputs = layers.Input(shape=input_shape)

    # Early layer (entropy monitoring target)
    x = layers.Conv2D(
        32, 3, padding="same", use_bias=False, name="early_conv"
    )(inputs)
    x = layers.BatchNormalization(name="early_bn")(x)
    x = layers.Activation("relu", name="early_act")(x)

    x = depthwise_block(x, 64, stride=2, name="block1")
    if use_attention:
        x = se_block(x, name="se1")

    x = depthwise_block(x, 128, stride=2, name="block2")
    if use_attention:
        x = se_block(x, name="se2")

    x = depthwise_block(x, 256, stride=2, name="block3")

    x = layers.GlobalAveragePooling2D(name="gap")(x)

    logits = layers.Dense(
        num_classes,
        name="logits"
    )(x)

    outputs = layers.Activation("softmax", name="softmax")(logits)

    model = models.Model(inputs, outputs, name="SimpleCNN")

    return model
