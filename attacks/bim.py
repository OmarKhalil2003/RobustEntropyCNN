import tensorflow as tf
from attacks.base_attack import BaseAttack

class BIMAttack(BaseAttack):
    """
    Basic Iterative Method (Iterative FGSM)
    """

    def __init__(
        self,
        model,
        eps=0.03,
        alpha=0.005,
        steps=10,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def generate(self, x, y):
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
