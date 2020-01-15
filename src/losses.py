import tensorflow as tf
import tensorflow.keras as tfk
from utils import iou


class ClassLoss(tfk.losses.Loss):
    def __init__(self, epsilon=1e-16):
        super(ClassLoss, self).__init__(
            reduction=tfk.losses.Reduction.SUM)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        num_objects = tf.math.count_nonzero(
            y_pred._keras_mask, dtype=tf.float32)

        diff = tf.math.reduce_sum(
            y_true * (-tf.math.log(y_pred + self.epsilon)) +
            (1 - y_true) * (-tf.math.log(1 - y_pred + self.epsilon)), -1)

        diff = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

        return diff / num_objects


class ConfidenceLoss(tfk.losses.Loss):
    def __init__(self, epsilon=1e-16):
        super(ConfidenceLoss, self).__init__(
            reduction=tfk.losses.Reduction.SUM)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        num_objects = tf.math.count_nonzero(
            y_pred._keras_mask, dtype=tf.float32)

        confidence, bboxes = tf.split(y_pred, [1, 4], axis=-1)

        ious = iou(y_true, bboxes, epsilon=self.epsilon)
        diff = tf.math.square(ious - tf.squeeze(confidence, -1)) / 20

        return diff / num_objects


class BboxLoss(tfk.losses.Loss):
    def __init__(self):
        super(BboxLoss, self).__init__(
            reduction=tfk.losses.Reduction.SUM)

    def call(self, y_true, y_pred):
        num_objects = tf.math.count_nonzero(
            y_pred._keras_mask, dtype=tf.float32)

        diff = tf.math.reduce_sum(tf.math.square(y_true - y_pred), -1)

        return diff / num_objects
