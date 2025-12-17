import tensorflow as tf

class DeepFoolAttack:
    """
    Simplified DeepFool (untargeted).
    Works per-batch for efficiency.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        max_iter: int = 20,
        overshoot: float = 0.02,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        self.model = model
        self.max_iter = max_iter
        self.overshoot = overshoot
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(self, x, y=None):
        x_adv = tf.identity(x)

        for _ in range(self.max_iter):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                logits = self.model(x_adv, training=False)

                preds = tf.argmax(logits, axis=1)
                correct_logits = tf.gather(
                    logits,
                    preds,
                    axis=1,
                    batch_dims=1
                )

            grads = tape.gradient(correct_logits, x_adv)
            perturbation = grads / (
                tf.norm(grads, ord=2, axis=[1, 2, 3], keepdims=True) + 1e-8
            )

            x_adv = x_adv + self.overshoot * perturbation
            x_adv = tf.clip_by_value(x_adv, self.clip_min, self.clip_max)

        return x_adv
