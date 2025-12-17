import matplotlib.pyplot as plt

def plot_selective_tradeoff():
    coverage = [1.0, 0.94, 0.85, 0.75, 0.65]
    accuracy = [0.66, 0.66, 0.70, 0.74, 0.78]

    plt.figure(figsize=(6, 5))
    plt.plot(coverage, accuracy, marker="o")

    plt.xlabel("Coverage (fraction of accepted samples)")
    plt.ylabel("Accuracy")
    plt.title("Selective Classification Trade-off")

    plt.grid(True)
    plt.tight_layout()
    plt.show()
