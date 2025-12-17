import tensorflow as tf

class BaseAttack:
    """
    Base class for adversarial attacks.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        loss_fn=None,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        self.model = model
        self.loss_fn = loss_fn or tf.keras.losses.SparseCategoricalCrossentropy()
        self.clip_min = clip_min
        self.clip_max = clip_max

    def clip(self, x):
        return tf.clip_by_value(x, self.clip_min, self.clip_max)

    def compute_gradient(self, x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = self.model(x, training=False)
            loss = self.loss_fn(y, preds)
        grad = tape.gradient(loss, x)
        return grad
