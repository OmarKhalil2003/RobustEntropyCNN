import tensorflow as tf

def selective_accuracy(
    model,
    entropy_monitor,
    threshold_detector,
    dataset
):
    kept_correct = tf.constant(0.0, dtype=tf.float32)
    kept_total = tf.constant(0.0, dtype=tf.float32)
    rejected = tf.constant(0.0, dtype=tf.float32)

    for x, y in dataset:
        entropy_dict = entropy_monitor.compute_batch_entropy(x)
        _, reject = threshold_detector.detect(entropy_dict)

        batch_size = tf.cast(tf.shape(y)[0], tf.float32)

        if reject:
            rejected += batch_size
            continue

        preds = tf.argmax(model(x, training=False), axis=1)
        y = tf.cast(y, tf.int64)

        correct = tf.reduce_sum(
            tf.cast(preds == y, tf.float32)
        )

        kept_correct += correct
        kept_total += batch_size

    coverage = kept_total / (kept_total + rejected + 1e-8)
    accuracy = kept_correct / (kept_total + 1e-8)

    return {
        "accuracy": float(accuracy.numpy()),
        "coverage": float(coverage.numpy())
    }
