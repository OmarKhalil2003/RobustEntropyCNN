import numpy as np

class EntropyThreshold:
    """
    Simple threshold-based entropy detector.
    """

    def __init__(self, method="mean_std", k=2.0):
        self.method = method
        self.k = k
        self.thresholds = {}

    def fit(self, entropy_records: dict):
        """
        entropy_records:
        {
            layer_name: [entropy_value_1, entropy_value_2, ...]
        }
        """

        for layer, values in entropy_records.items():
            values = np.array(values)

            if self.method == "mean_std":
                mu = values.mean()
                sigma = values.std()
                self.thresholds[layer] = mu + self.k * sigma

            elif self.method == "percentile":
                self.thresholds[layer] = np.percentile(values, 95)

            else:
                raise ValueError("Unknown thresholding method")

    def detect(self, entropy_dict: dict):
        """
        Returns:
        - decision per layer
        - overall decision (any layer exceeds threshold)
        """

        layer_flags = {}
        overall_flag = False

        for layer, entropy in entropy_dict.items():
            threshold = self.thresholds.get(layer)
            flag = entropy > threshold
            layer_flags[layer] = bool(flag)
            overall_flag = overall_flag or flag

        return layer_flags, overall_flag
