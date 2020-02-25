import tensorflow as tf
import tensorflow.keras as tfk

from squeezedet.utils import iou


class AveragePrecision(tfk.metrics.Metric):
    def __init__(self, **kwargs):
        super(AveragePrecision, self).__init__(**kwargs)
        self.precision = self.add_weight(
            name='precision',
            initializer='zeros',
            aggregation=tf.VariableAggregation.MEAN)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.assign_add(tf.math.reduce_sum(y_true))

    def result(self):
        return self.precision


class AverageRecall(tfk.metrics.Metric):
    def __init__(self, **kwargs):
        super(AverageRecall, self).__init__(**kwargs)
        self.recall = self.add_weight(
            name='recall',
            initializer='zeros',
            aggregation=tf.VariableAggregation.MEAN)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.recall.assign_add(tf.math.reduce_sum(y_true))

    def result(self):
        return self.recall
