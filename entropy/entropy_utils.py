import tensorflow as tf
import numpy as np

def compute_entropy(
    activations: tf.Tensor,
    num_bins: int = 30,
    eps: float = 1e-8
) -> tf.Tensor:
    """
    Computes Shannon entropy of activations.
    - ReLU applied (only positive activations)
    - Histogram-based estimation
    - Batch-level entropy
    """

    # Flatten activations
    x = tf.reshape(activations, [-1])

    # Keep positive values (post-ReLU behavior)
    x = tf.nn.relu(x)

    # Avoid empty tensors
    max_val = tf.reduce_max(x)
    max_val = tf.maximum(max_val, eps)

    hist = tf.histogram_fixed_width(
        x,
        value_range=[0.0, max_val],
        nbins=num_bins
    )

    hist = tf.cast(hist, tf.float32)
    prob = hist / (tf.reduce_sum(hist) + eps)

    prob = tf.boolean_mask(prob, prob > 0)

    entropy = -tf.reduce_sum(prob * tf.math.log(prob + eps))

    return entropy

def selective_accuracy(entropy, y_true, y_pred, thresholds):
    """
    Selective accuracy vs coverage based on entropy rejection.
    """

    selective_acc = []
    coverage = []

    N = len(entropy)

    for tau in thresholds:
        mask = entropy <= tau
        accepted = np.sum(mask)

        if accepted == 0:
            continue

        cov = accepted / N
        acc = np.mean(y_pred[mask] == y_true[mask])

        coverage.append(cov)
        selective_acc.append(acc)

    return np.array(coverage), np.array(selective_acc)


def rejection_rate(entropy, threshold):
    """
    Fraction of rejected samples.
    """
    rejected = np.sum(entropy > threshold)
    return rejected / len(entropy)

def softmax_entropy_from_logits(logits, eps=1e-8):
    """
    Computes per-sample softmax entropy from logits.
    logits: (N, C)
    returns: (N,)
    """
    probs = tf.nn.softmax(logits, axis=1)
    return -tf.reduce_sum(probs * tf.math.log(probs + eps), axis=1)
