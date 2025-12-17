from models.base_models.simple_cnn import build_simple_cnn
from models.base_models.mobilenet_custom import build_mobilenet_custom
from models.base_models.resnet_custom import build_resnet_custom
from models.base_models.hybrid_cnn import build_hybrid_cnn


def build_model(
    model_name="simple_cnn",
    input_shape=(96, 96, 3),
    num_classes=10,
    **kwargs
):
    if model_name == "simple_cnn":
        return build_simple_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs
        )

    elif model_name == "mobilenet_custom":
        return build_mobilenet_custom(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs
        )

    elif model_name == "resnet_custom":
        return build_resnet_custom(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs
        )

    elif model_name == "hybrid_cnn":
        return build_hybrid_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")
