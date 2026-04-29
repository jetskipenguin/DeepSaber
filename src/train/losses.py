import tensorflow as tf

def calculate_perplexity(y_true, y_pred, from_logits=False, label_smoothing=0.0):
    """
    Based on Gregorgeous github Gist: https://gist.github.com/Gregorgeous/dbad1ec22efc250c76354d949a13cec3
    Does not support masks.
    """
    # The next 4 lines zero-out the padding from loss calculations,
    # this follows the logic from: https://www.tensorflow.org/beta/tutorials/text/transformer#loss_and_metrics
    # mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, from_logits=from_logits, label_smoothing=label_smoothing
    )
    step1 = tf.reduce_mean(loss_, axis=-1)
    step2 = tf.exp(step1)
    perplexity = tf.reduce_mean(step2)
    return perplexity

class Perplexity(tf.keras.losses.Loss):
    def __init__(self,
                 from_logits=False,
                 label_smoothing=0.0,
                 reduction='sum_over_batch_size',
                 name='perplexity'):
        """Initializes `Perplexity` instance.

        Args:
          from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
            **Note - Using from_logits=True is more numerically stable.**
          label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. e.g.
            `label_smoothing=0.2` means that we will use a value of `0.1` for label
            `0` and `0.9` for label `1`"
          reduction: Type of `tf.keras.losses.Reduction` to apply to loss. 
          name: Optional name for the op. Defaults to 'perplexity'.
        """
        super(Perplexity, self).__init__(name=name, reduction=reduction)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        return calculate_perplexity(
            y_true, y_pred, 
            from_logits=self.from_logits, 
            label_smoothing=self.label_smoothing
        )