import tensorflow as tf

class CarliniWagnerL2:
    """
    Simplified Carlini & Wagner L2 attack.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        c: float = 1.0,
        lr: float = 0.01,
        steps: int = 100,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        self.model = model
        self.c = c
        self.lr = lr
        self.steps = steps
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(self, x, y):
        x_adv = tf.Variable(tf.identity(x))
        optimizer = tf.keras.optimizers.Adam(self.lr)

        y_onehot = tf.one_hot(y, depth=self.model.output_shape[-1])

        for _ in range(self.steps):
            with tf.GradientTape() as tape:
                logits = self.model(x_adv, training=False)

                real = tf.reduce_sum(y_onehot * logits, axis=1)
                other = tf.reduce_max(
                    (1 - y_onehot) * logits - y_onehot * 1e4,
                    axis=1
                )

                loss_cls = tf.maximum(real - other, 0.0)
                loss_l2 = tf.reduce_sum(tf.square(x_adv - x), axis=[1, 2, 3])

                loss = tf.reduce_mean(loss_l2 + self.c * loss_cls)

            grads = tape.gradient(loss, x_adv)
            optimizer.apply_gradients([(grads, x_adv)])

            x_adv.assign(tf.clip_by_value(x_adv, self.clip_min, self.clip_max))

        return x_adv
