import matplotlib.pyplot as plt

def plot_accuracy_comparison():
    labels = [
        "Clean",
        "FGSM",
        "PGD",
        "Selective\n(Entropy)"
    ]

    accuracies = [
        0.6577,   # clean
        0.0255,   # FGSM
        0.0000,   # PGD
        0.6553    # selective
    ]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, accuracies)

    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Under Adversarial Attacks")

    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y + 0.02,
            f"{y:.2f}",
            ha="center"
        )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
