import tensorflow as tf
from attacks.base_attack import BaseAttack

class PGDAttack(BaseAttack):
    """
    Projected Gradient Descent (PGD)
    """

    def __init__(
        self,
        model,
        eps=0.03,
        alpha=0.005,
        steps=20,
        random_start=True,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def generate(self, x, y):
        if self.random_start:
            x_adv = x + tf.random.uniform(
                tf.shape(x),
                -self.eps,
                self.eps
            )
            x_adv = self.clip(x_adv)
        else:
            x_adv = tf.identity(x)

        for _ in range(self.steps):
            grad = self.compute_gradient(x_adv, y)
            x_adv = x_adv + self.alpha * tf.sign(grad)
            x_adv = tf.clip_by_value(
                x_adv,
                x - self.eps,
                x + self.eps
            )
            x_adv = self.clip(x_adv)

        return x_adv
