import tensorflow as tf


class L2Regularizer:
    def __init__(self, scaling: float):
        self.scaling = scaling

    def get_penalty(self, weights: tf.Tensor) -> tf.Tensor:
        return self.scaling * tf.math.sqrt(tf.nn.l2_loss(t=weights, name="l2RegularizationLoss"))
