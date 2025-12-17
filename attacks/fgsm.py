import tensorflow as tf
from attacks.base_attack import BaseAttack

class FGSMAttack(BaseAttack):
    """
    Fast Gradient Sign Method
    """

    def __init__(self, model, eps=0.03, **kwargs):
        super().__init__(model, **kwargs)
        self.eps = eps

    def generate(self, x, y):
        grad = self.compute_gradient(x, y)
        x_adv = x + self.eps * tf.sign(grad)
        return self.clip(x_adv)
