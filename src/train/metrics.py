import tensorflow as tf
from train.losses import calculate_perplexity
from utils.types import Config

def create_metrics(is_train, config: Config):
    if is_train:
        return []  # all are computed by the AVS model
    return ['acc', Perplexity()]

def compute_acc(res_dict):
    acc = [val for key, val in res_dict.items() if 'acc' in key]
    if len(acc) == 0:
        return 0.0
    return sum(acc) / len(acc)

class CosineDistance(tf.keras.metrics.Mean):
    def __init__(self, name='cosine_distance', dtype=None):
        """Creates a `CosineSimilarity` instance.

        Args:
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(CosineDistance, self).__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred = tf.nn.l2_normalize(y_pred, axis=-1)
        cos_sim = tf.reduce_sum(y_true * y_pred, axis=-1)
        return super(CosineDistance, self).update_state(1.0 - cos_sim, sample_weight)

class Perplexity(tf.keras.metrics.Mean):
    def __init__(self, name='perplexity', dtype=None):
        super(Perplexity, self).__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        perplexity = calculate_perplexity(y_true, y_pred)
        return super(Perplexity, self).update_state(perplexity, sample_weight)