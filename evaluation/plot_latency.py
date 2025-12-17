import matplotlib.pyplot as plt

def plot_latency():
    labels = ["Model Only", "Model + Entropy"]
    latency = [0.095, 0.120]  # seconds (approx)

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, latency)

    plt.ylabel("Latency (seconds)")
    plt.title("Inference Latency Overhead")

    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y + 0.002,
            f"{y:.3f}s",
            ha="center"
        )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
