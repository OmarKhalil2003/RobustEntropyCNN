import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from utils.seed import set_seed
from utils.gpu import setup_gpu
from data.load_data import load_cifar10
from models.build_model import build_model
from entropy.entropy_monitor import EntropyMonitor
from entropy.thresholding import EntropyThreshold
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack
from evaluation.robustness_metrics import (
    evaluate_accuracy,
    evaluate_under_attack
)
from evaluation.selective_classification import selective_accuracy
from evaluation.latency_benchmark import benchmark_latency


def main():
    set_seed(42)
    setup_gpu()

    # Load data
    train_ds, val_ds, test_ds = load_cifar10(batch_size=128)

    # Build model
    model = build_model(
        model_name="hybrid_cnn",
        input_shape=(32, 32, 3),
        mode="2d",
        pretrained=True,
        freeze_backbone=True
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    print("Training model...")
    model.fit(train_ds, validation_data=val_ds, epochs=30)
    model.save("saved_model_hybrid.keras")

    # Baseline accuracy
    clean_acc = evaluate_accuracy(model, test_ds)
    print(f"Clean accuracy: {clean_acc:.4f}")

    # Entropy monitoring
    monitor = EntropyMonitor(
        model,
        layer_names=["early_conv", "logits"],
        num_bins=30
    )

    # Collect clean entropy
    entropy_records = {"early_conv": [], "logits": []}
    for x, _ in val_ds:
        ent = monitor.compute_batch_entropy(x)
        for k in entropy_records:
            entropy_records[k].append(float(ent[k]))

    threshold = EntropyThreshold(method="mean_std", k=2.0)
    threshold.fit(entropy_records)

    # Attacks
    fgsm = FGSMAttack(model, eps=0.03)
    pgd = PGDAttack(model, eps=0.03, steps=20)

    fgsm_acc = evaluate_under_attack(model, fgsm, test_ds)
    pgd_acc = evaluate_under_attack(model, pgd, test_ds)

    print(f"FGSM accuracy: {fgsm_acc:.4f}")
    print(f"PGD accuracy: {pgd_acc:.4f}")

    # Selective classification
    selective = selective_accuracy(
        model,
        monitor,
        threshold,
        test_ds
    )

    print("Selective classification:")
    print(selective)

    # Latency
    latency = benchmark_latency(model, monitor, test_ds)
    print(f"Average latency (model + entropy): {latency:.4f} sec")


if __name__ == "__main__":
    main()
