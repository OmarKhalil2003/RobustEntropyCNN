import tensorflow as tf
from tensorflow.keras import layers, models

from models.layers.depthwise_blocks import depthwise_block
from models.layers.grouped_conv import grouped_conv_block
from models.layers.attention_se import se_block


def build_hybrid_cnn(
    input_shape=(32, 32, 3),
    num_classes=10,
    mode="2d",                 # "2d" | "3d"
    use_grouped=True,
    groups=4,
    use_attention=True,
    pretrained=True,
    freeze_backbone=True
):
    """
    Phase 2 — Hybrid Entropy CNN

    Features:
    - Transfer learning backbone (MobileNetV2)
    - Built-in preprocessing & resizing (model-level)
    - Grouped / depthwise hybrid blocks
    - SE attention
    - Optional 3D pathway for medical imaging
    - Fully compatible with entropy monitoring
    """

    # ======================================================
    # =============== 2D TRANSFER LEARNING =================
    # ======================================================
    if mode == "2d":

        # -------- Input --------
        inputs = layers.Input(shape=input_shape, name="input")

        # -------- Resize (dataset-agnostic) --------
        x = layers.Resizing(96, 96, name="resize")(inputs)

        # -------- MobileNet preprocessing --------
        x = layers.Lambda(
            lambda t: tf.keras.applications.mobilenet_v2.preprocess_input(t),
            name="mobilenet_preprocess"
        )(x)

        # -------- Pretrained backbone --------
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(96, 96, 3),
            include_top=False,
            weights="imagenet" if pretrained else None
        )

        if freeze_backbone:
            backbone.trainable = False

        x = backbone(x)

        # -------- Early entropy-monitored layer --------
        x = layers.Conv2D(
            64,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="early_conv"
        )(x)
        x = layers.BatchNormalization(name="early_bn")(x)
        x = layers.Activation("relu", name="early_act")(x)

        if use_attention:
            x = se_block(x, name="se_adapt")

        # -------- Hybrid feature extractor --------
        if use_grouped:
            x = grouped_conv_block(
                x,
                filters=128,
                groups=groups,
                stride=2,
                name="grouped_block"
            )
        else:
            x = depthwise_block(
                x,
                pointwise_filters=128,
                stride=2,
                name="depthwise_block"
            )

        if use_attention:
            x = se_block(x, name="se_hybrid")

        x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ======================================================
    # =================== 3D PATHWAY =======================
    # ======================================================
    elif mode == "3d":

        inputs = layers.Input(shape=input_shape, name="input")

        x = layers.Conv3D(
            32,
            kernel_size=3,
            padding="same",
            activation="relu",
            name="early_conv"
        )(inputs)
        x = layers.MaxPool3D()(x)

        x = layers.Conv3D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPool3D()(x)

        x = layers.Conv3D(128, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling3D()(x)

    else:
        raise ValueError("mode must be '2d' or '3d'")

    # ======================================================
    # ================== CLASSIFIER ========================
    # ======================================================
    logits = layers.Dense(num_classes, name="logits")(x)
    outputs = layers.Activation("softmax", name="softmax")(logits)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="HybridEntropyCNN"
    )

    return model
