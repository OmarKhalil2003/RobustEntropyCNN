import tensorflow as tf
from tqdm import tqdm

def evaluate_accuracy(model, dataset):
    correct = 0
    total = 0

    for x, y in dataset:
        preds = tf.argmax(model(x, training=False), axis=1)
        y = tf.cast(y, tf.int64)
        correct += tf.reduce_sum(tf.cast(preds == y, tf.int32))
        total += y.shape[0]

    return float(correct) / float(total)


def evaluate_under_attack(model, attack, dataset):
    correct = 0
    total = 0

    for x, y in tqdm(dataset, desc="Evaluating under attack"):
        x_adv = attack.generate(x, y)
        preds = tf.argmax(model(x_adv, training=False), axis=1)
        y = tf.cast(y, tf.int64)
        correct += tf.reduce_sum(tf.cast(preds == y, tf.int32))
        total += y.shape[0]

    return float(correct) / float(total)
