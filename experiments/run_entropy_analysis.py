import tensorflow as tf

from data.load_data import load_cifar10
from models.build_model import build_model
from entropy.entropy_monitor import EntropyMonitor
from attacks.fgsm import FGSMAttack
from evaluation.visualization import plot_entropy_distributions
from utils.seed import set_seed
from utils.gpu import setup_gpu


def main():
    set_seed(42)
    setup_gpu()

    _, val_ds, test_ds = load_cifar10(batch_size=128)

    model = build_model("simple_cnn", use_attention=True)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy"
    )

    # Train quickly (analysis needs a trained model)
    model.fit(val_ds, epochs=30)

    monitor = EntropyMonitor(
        model,
        layer_names=["early_conv", "softmax"]
    )

    clean_entropy = {"early_conv": [], "softmax": []}
    adv_entropy = {"early_conv": [], "softmax": []}

    fgsm = FGSMAttack(model, eps=0.03)

    for x, y in test_ds:
        clean_e = monitor.compute_batch_entropy(x)
        for k in clean_entropy:
            clean_entropy[k].append(float(clean_e[k]))

        x_adv = fgsm.generate(x, y)
        adv_e = monitor.compute_batch_entropy(x_adv)
        for k in adv_entropy:
            adv_entropy[k].append(float(adv_e[k]))

    plot_entropy_distributions(
        clean_entropy,
        adv_entropy,
        title_prefix="FGSM"
    )


if __name__ == "__main__":
    main()
