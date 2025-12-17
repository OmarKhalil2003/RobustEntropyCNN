import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from data.load_data import load_cifar10
from models.build_model import build_model
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack
from attacks.bim import BIMAttack
from evaluation.robustness_metrics import evaluate_under_attack
from utils.seed import set_seed
from utils.gpu import setup_gpu


def main():
    set_seed(42)
    setup_gpu()

    train_ds, val_ds, test_ds = load_cifar10(batch_size=128)

    model = build_model("simple_cnn", use_attention=True)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=5)

    attacks = {
        "FGSM": FGSMAttack(model, eps=0.03),
        "BIM": BIMAttack(model, eps=0.03, steps=10),
        "PGD": PGDAttack(model, eps=0.03, steps=20),
    }

    for name, attack in attacks.items():
        acc = evaluate_under_attack(model, attack, test_ds)
        print(f"{name} accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
