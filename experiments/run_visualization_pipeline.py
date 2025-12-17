import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
from entropy.entropy_utils import selective_accuracy, rejection_rate
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from data.load_data import load_cifar10
from entropy.entropy_monitor import EntropyMonitor
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack

# ----------------------------
# Setup
# ----------------------------
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

def softmax_entropy_per_sample(probs, eps=1e-8):
    """
    probs: (N, C) softmax probabilities
    returns: (N,) entropy per sample
    """
    return -np.sum(probs * np.log(probs + eps), axis=1)
# ----------------------------
# Load model & data
# ----------------------------
_, _, test_ds = load_cifar10(batch_size=128)
model = tf.keras.models.load_model("saved_model_hybrid.keras",safe_mode=False)

monitor = EntropyMonitor(model, layer_names=["softmax"])

fgsm = FGSMAttack(model, eps=0.03)
pgd = PGDAttack(model, eps=0.03, steps=20)

y_true = []
y_clean = []
y_fgsm = []
y_pgd = []
entropy_clean = []
entropy_fgsm = []
confidence_fgsm = []

# ----------------------------
# Collect REAL data
# ----------------------------
for x, y in test_ds:
    preds_clean = tf.argmax(model(x, training=False), axis=1)

    x_fgsm = fgsm.generate(x, y)
    preds_fgsm = tf.argmax(model(x_fgsm, training=False), axis=1)

    x_pgd = pgd.generate(x, y)
    preds_pgd = tf.argmax(model(x_pgd, training=False), axis=1)

    ent_clean = monitor.compute_batch_entropy(x)["softmax"]
    ent_fgsm = monitor.compute_batch_entropy(x_fgsm)["softmax"]

    y_true.extend(y.numpy())
    y_clean.extend(preds_clean.numpy())
    y_fgsm.extend(preds_fgsm.numpy())
    y_pgd.extend(preds_pgd.numpy())
    probs_clean = tf.nn.softmax(model(x, training=False)).numpy()
    probs_fgsm = tf.nn.softmax(model(x_fgsm, training=False)).numpy()
    entropy_fgsm.extend(softmax_entropy_per_sample(probs_fgsm))
    confidence_fgsm.extend(np.max(probs_fgsm, axis=1))
    entropy_clean.extend(softmax_entropy_per_sample(probs_clean))

y_true = np.array(y_true)
y_clean = np.array(y_clean)
y_fgsm = np.array(y_fgsm)
y_pgd = np.array(y_pgd)
entropy_clean = np.array(entropy_clean)
entropy_fgsm = np.array(entropy_fgsm)
confidence_fgsm = np.array(confidence_fgsm)

# ----------------------------
# Accuracy table (REAL)
# ----------------------------
acc_clean = np.mean(y_clean == y_true)
acc_fgsm = np.mean(y_fgsm == y_true)
acc_pgd = np.mean(y_pgd == y_true)

acc_table = pd.DataFrame({
    "Setting": ["Clean", "FGSM", "PGD"],
    "Accuracy": [acc_clean, acc_fgsm, acc_pgd]
})
acc_table.to_csv("results/tables/accuracy_comparison.csv", index=False)

# ----------------------------
# Figure 1 — Accuracy comparison
# ----------------------------
plt.figure(figsize=(7,5))
plt.bar(acc_table["Setting"], acc_table["Accuracy"])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Accuracy Under Adversarial Attacks")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/accuracy_comparison.png", dpi=300)
plt.close()

# ----------------------------
# Figure 2 — Entropy distributions
# ----------------------------
plt.figure(figsize=(7,5))
plt.hist(entropy_clean, bins=30, alpha=0.7, label="Clean", density=True)
plt.hist(entropy_fgsm, bins=30, alpha=0.7, label="FGSM", density=True)
plt.xlabel("Softmax Entropy")
plt.ylabel("Density")
plt.title("Softmax Entropy Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/entropy_distribution.png", dpi=300)
plt.close()

# ----------------------------
# Figure 3 — Selective accuracy vs coverage (REAL & FIXED)
# ----------------------------
thresholds = np.linspace(
    entropy_fgsm.min(),
    entropy_fgsm.max(),
    30
)

coverages, accuracies = selective_accuracy(
    entropy=entropy_fgsm,
    y_true=y_true,
    y_pred=y_fgsm,
    thresholds=thresholds
)

tradeoff = pd.DataFrame({
    "Coverage": coverages,
    "Accuracy": accuracies
})
tradeoff.to_csv("results/tables/selective_tradeoff.csv", index=False)

plt.figure(figsize=(6,5))
plt.plot(coverages, accuracies, marker="o")
plt.xlabel("Coverage")
plt.ylabel("Accuracy")
plt.title("Selective Accuracy vs Coverage")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/selective_tradeoff.png", dpi=300)
plt.close()
# ----------------------------
# Figure 4 — Entropy vs correctness
# ----------------------------
correct_mask = (y_fgsm == y_true)

plt.figure(figsize=(6,5))
plt.scatter(
    entropy_fgsm[correct_mask],
    np.ones(correct_mask.sum()),
    alpha=0.4,
    label="Correct"
)
plt.scatter(
    entropy_fgsm[~correct_mask],
    np.zeros((~correct_mask).sum()),
    alpha=0.4,
    label="Incorrect"
)
plt.yticks([0,1], ["Wrong", "Correct"])
plt.xlabel("Softmax Entropy")
plt.title("Entropy vs Prediction Correctness (FGSM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/entropy_vs_correctness.png", dpi=300)
plt.close()
# ----------------------------
# Figure 5 — Confusion Matrices
# ----------------------------
def save_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, cmap="Blues", square=True)
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"results/figures/{name}.png", dpi=300)
    plt.close()
save_confusion_matrix(y_true, y_clean, "confusion_clean")
save_confusion_matrix(y_true, y_fgsm, "confusion_fgsm")

# ----------------------------
# Figure 6 — Accuracy vs FGSM ε (Robustness Curve)
# ----------------------------
epsilons = [0.005, 0.01, 0.02, 0.03, 0.05]
acc_eps = []

for eps in epsilons:
    fgsm_eps = FGSMAttack(model, eps=eps)
    preds = []
    for x, y in test_ds:
        x_adv = fgsm_eps.generate(x, y)
        preds.extend(tf.argmax(model(x_adv), axis=1).numpy())
    acc_eps.append(np.mean(np.array(preds) == y_true))
plt.figure(figsize=(6,5))
plt.plot(epsilons, acc_eps, marker="o")
plt.xlabel("FGSM ε")
plt.ylabel("Accuracy")
plt.title("Accuracy Degradation vs FGSM Strength")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/accuracy_vs_epsilon.png", dpi=300)
plt.close()
# ----------------------------
# Figure 6 — Entropy vs Confidence
# ----------------------------
confidence_fgsm = np.array(confidence_fgsm)

print("Entropy FGSM:", entropy_fgsm.shape)
print("Confidence FGSM:", confidence_fgsm.shape)
plt.figure(figsize=(6,5))
plt.scatter(confidence_fgsm, entropy_fgsm, alpha=0.4)
plt.xlabel("Max Softmax Probability")
plt.ylabel("Entropy")
plt.title("Entropy vs Confidence (FGSM)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/entropy_vs_confidence.png", dpi=300)
plt.close()
# ----------------------------
# Figure 7 — Rejection Rate per Attack
# ----------------------------
tau = np.percentile(entropy_clean, 70)  # calibration threshold

rejection = {
    "Clean": rejection_rate(entropy_clean, tau),
    "FGSM": rejection_rate(entropy_fgsm, tau),
}

plt.figure(figsize=(5,4))
plt.bar(rejection.keys(), rejection.values())
plt.ylabel("Rejection Rate")
plt.title("Rejection Rate by Input Type")
plt.tight_layout()
plt.savefig("results/figures/rejection_rate.png", dpi=300)
plt.close()
# ----------------------------
# Figure 8 — Layer-wise Entropy Comparison
# ----------------------------
monitor_layers = EntropyMonitor(model, ["early_conv", "softmax"])

ent_early = []
ent_soft = []

for x, _ in test_ds:
    e = monitor_layers.compute_batch_entropy(x)
    ent_early.append(float(e["early_conv"]))
    ent_soft.append(float(e["softmax"]))

plt.figure(figsize=(6,5))
plt.boxplot([ent_early, ent_soft], labels=["Early Conv", "Softmax"])
plt.ylabel("Entropy")
plt.title("Layer-wise Entropy Comparison (Clean)")
plt.tight_layout()
plt.savefig("results/figures/layerwise_entropy.png", dpi=300)
plt.close()



print("All visualizations and tables saved to /results")
