import tensorflow as tf
from entropy.entropy_utils import compute_entropy

class EntropyMonitor:
    """
    Non-invasive entropy monitor for CNNs.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        layer_names: list,
        num_bins: int = 30
    ):
        self.layer_names = layer_names
        self.num_bins = num_bins

        # Multi-output model (no graph modification)
        self.monitor_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(name).output for name in layer_names]
        )

    def compute_batch_entropy(self, x: tf.Tensor):
        """
        Computes entropy for each monitored layer.
        """
        activations = self.monitor_model(x, training=False)

        if not isinstance(activations, list):
            activations = [activations]

        entropies = {}

        for name, act in zip(self.layer_names, activations):
            entropies[name] = compute_entropy(
                act,
                num_bins=self.num_bins
            )

        return entropies
