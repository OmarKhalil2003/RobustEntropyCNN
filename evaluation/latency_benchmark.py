import time
import tensorflow as tf

def benchmark_latency(model, entropy_monitor, dataset, warmup=5, runs=20):
    # Warmup
    for i, (x, _) in enumerate(dataset):
        model(x, training=False)
        entropy_monitor.compute_batch_entropy(x)
        if i >= warmup:
            break

    times = []

    for i, (x, _) in enumerate(dataset):
        start = time.time()
        _ = model(x, training=False)
        _ = entropy_monitor.compute_batch_entropy(x)
        times.append(time.time() - start)

        if i >= runs:
            break

    return sum(times) / len(times)
