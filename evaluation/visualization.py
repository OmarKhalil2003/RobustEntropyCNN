import matplotlib.pyplot as plt
import numpy as np

def plot_entropy_distributions(
    clean_entropy: dict,
    adv_entropy: dict,
    title_prefix: str = ""
):
    """
    clean_entropy / adv_entropy:
    {
        layer_name: [entropy_values]
    }
    """

    layers = clean_entropy.keys()

    for layer in layers:
        plt.figure(figsize=(6, 4))

        plt.hist(
            clean_entropy[layer],
            bins=30,
            alpha=0.7,
            label="Clean",
            density=True
        )

        plt.hist(
            adv_entropy[layer],
            bins=30,
            alpha=0.7,
            label="Adversarial",
            density=True
        )

        plt.title(f"{title_prefix} Entropy Distribution — {layer}")
        plt.xlabel("Entropy")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
