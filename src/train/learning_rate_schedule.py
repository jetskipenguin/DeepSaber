import math
import tensorflow as tf

class FlatCosAnnealSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a flat cosine decay schedule.

    See fastAI discussion https://forums.fast.ai/t/fastai-v2-callbacks-learner-optimizer/53519
    Updated for Keras 3 / TF 2.16+ API.

    Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
    """

    def __init__(
            self,
            decay_start,
            initial_learning_rate,
            decay_steps,
            alpha=0.0,
            name=None):
        """If Applies cosine decay to the learning rate.

        Args:
          decay_start: A scalar `int32` or `int64` `Tensor` or a Python number.
            Number of steps to wait before performing cosine annealing.
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a
            Python number. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Number of steps to decay over.
          alpha: A scalar `float32` or `float64` Tensor or a Python number.
            Minimum learning rate value as a fraction of initial_learning_rate.
          name: String. Optional name of the operation.  Defaults to 'CosineDecay'.
        """
        super(FlatCosAnnealSchedule, self).__init__()
        self.decay_start = tf.cast(decay_start, tf.float32)
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.alpha = tf.cast(alpha, tf.float32)
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "FlatCosAnnealSchedule"):
            step = tf.cast(step, tf.float32)
            decay_start = self.decay_start
            decay_steps = self.decay_steps - decay_start

            global_step_recomp = step - decay_start
            global_step_recomp = tf.maximum(0.0, global_step_recomp)
            global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
            completed_fraction = global_step_recomp / decay_steps
            cosine_decayed = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * completed_fraction))

            decayed = (1.0 - self.alpha) * cosine_decayed + self.alpha
            return self.initial_learning_rate * decayed

    def get_config(self):
        return {
            "decay_start": float(self.decay_start),
            "initial_learning_rate": float(self.initial_learning_rate),
            "decay_steps": float(self.decay_steps),
            "alpha": float(self.alpha),
            "name": self.name
        }