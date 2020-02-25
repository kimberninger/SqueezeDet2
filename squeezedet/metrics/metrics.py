import tensorflow as tf
import tensorflow.keras as tfk

from ..utils import iou


def true_and_false_positives(bboxes_true, labels_true,
                             bboxes_pred, labels_pred, confidence,
                             num_classes, iou_threshold=0.5):
    order = tf.argsort(
        tf.where(labels_pred >= 0, confidence, 0.0),
        direction='DESCENDING')

    ious = iou(
        bboxes_true[..., tf.newaxis, :, :],
        bboxes_pred[..., tf.newaxis, :])

    class_indices = tf.reshape(
        tf.range(num_classes),
        shape=(1, -1, 1, 1))

    mask = tf.math.logical_and(
        tf.math.equal(
            labels_pred[:, tf.newaxis, ..., tf.newaxis],
            class_indices),
        tf.math.equal(
            labels_true[:, tf.newaxis, ..., tf.newaxis, :],
            class_indices))

    sorted_ious = tf.gather(
        tf.where(mask, ious[:, tf.newaxis], 0.0),
        tf.tile(
            order[:, tf.newaxis],
            multiples=(1, num_classes, 1)),
        batch_dims=2)

    assignments = tf.transpose(
        tf.scan(
            tf.math.logical_or,
            iou_threshold < tf.transpose(
                sorted_ious,
                perm=(2, 0, 1, 3))),
        perm=(1, 2, 0, 3))

    unique_assignments = tf.concat([
        assignments[..., :1, :],
        tf.math.logical_xor(
            assignments[..., 1:, :],
            assignments[..., :-1, :])
    ], axis=-2)

    tp = tf.gather(
        tf.math.reduce_any(
            unique_assignments,
            axis=-1),
        tf.tile(
            tf.argsort(order)[:, tf.newaxis],
            multiples=(1, num_classes, 1)),
        batch_dims=2)

    fp = tf.math.logical_and(
        tf.math.logical_not(tp),
        tf.math.reduce_any(mask, axis=-1))

    return tp, fp


def pr_curve(bboxes_true, labels_true,
             bboxes_pred, labels_pred, confidence,
             num_classes, iou_threshold=0.5):
    tp, fp = true_and_false_positives(
        bboxes_true, labels_true,
        bboxes_pred, labels_pred, confidence,
        num_classes, iou_threshold)

    order = tf.argsort(
        confidence[labels_pred >= 0],
        direction='DESCENDING')

    tp_acc = tf.cumsum(
        tf.cast(
            tf.gather(
                tf.transpose(tp, perm=(0, 2, 1))[labels_pred >= 0],
                order),
            dtype=tf.float32))

    fp_acc = tf.cumsum(
        tf.cast(
            tf.gather(
                tf.transpose(fp, perm=(0, 2, 1))[labels_pred >= 0],
                order),
            dtype=tf.float32))

    num_gt = tf.math.reduce_sum(
        tf.math.reduce_sum(
            tf.cast(
                tf.math.equal(
                    labels_true[:, tf.newaxis],
                    tf.range(num_classes)[tf.newaxis, :, tf.newaxis]),
                dtype=tf.float32),
            axis=-1),
        axis=0)

    precision = tf.math.divide_no_nan(tp_acc, tp_acc + fp_acc)
    recall = tf.math.divide_no_nan(tp_acc, num_gt)

    return precision, recall


def mean_average_precision(bboxes_true, labels_true,
                           bboxes_pred, labels_pred, confidence,
                           num_classes, iou_threshold=0.5,
                           num_thresholds=11):
    thresholds = tf.linspace(0., 1., num_thresholds)

    precision, recall = pr_curve(
        bboxes_true, labels_true,
        bboxes_pred, labels_pred, confidence,
        num_classes, iou_threshold)

    return tf.math.reduce_mean(
        tf.math.reduce_max(
            precision[..., tf.newaxis] * tf.cast(
                tf.math.greater_equal(
                    recall[..., tf.newaxis],
                    thresholds[tf.newaxis, tf.newaxis]), dtype=tf.float32),
            axis=0))


class AveragePrecision(tfk.metrics.Metric):
    def __init__(self, num_classes, iou_threshold=0.5, **kwargs):
        super(AveragePrecision, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.sum_precision = self.add_weight(
            name='sum_precision', initializer='zeros')
        self.num_samples = self.add_weight(
            name='num_samples', initializer='zeros')

    def update_state(self,
                     bboxes_true, labels_true,
                     bboxes_pred, labels_pred, confidence,
                     sample_weight=None):

        tp, fp = true_and_false_positives(
            bboxes_true, labels_true,
            bboxes_pred, labels_pred, confidence,
            self.num_classes, self.iou_threshold)

        tp = tf.math.reduce_sum(
            tf.math.reduce_sum(
                tf.cast(tp, dtype=tf.float32),
                axis=-1),
            axis=-1)

        fp = tf.math.reduce_sum(
            tf.math.reduce_sum(
                tf.cast(fp, dtype=tf.float32),
                axis=-1),
            axis=-1)

        precision = tf.math.divide_no_nan(tp, tp + fp)
        precision = tf.cast(precision, dtype=self.dtype)

        n = tf.ones_like(precision)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, precision)
            precision = tf.multiply(precision, sample_weight)
            n = tf.multiply(n, sample_weight)

        self.sum_precision.assign_add(tf.reduce_sum(precision))
        self.num_samples.assign_add(tf.reduce_sum(n))

    def result(self):
        return tf.math.divide_no_nan(
            self.sum_precision,
            self.num_samples)


class AverageRecall(tfk.metrics.Metric):
    def __init__(self, num_classes, iou_threshold=0.5, **kwargs):
        super(AverageRecall, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.sum_recall = self.add_weight(
            name='sum_precision', initializer='zeros')
        self.num_samples = self.add_weight(
            name='num_samples', initializer='zeros')

    def update_state(self,
                     bboxes_true, labels_true,
                     bboxes_pred, labels_pred, confidence,
                     sample_weight=None):

        tp, _ = true_and_false_positives(
            bboxes_true, labels_true,
            bboxes_pred, labels_pred, confidence,
            self.num_classes, self.iou_threshold)

        tp = tf.math.reduce_sum(
            tf.math.reduce_sum(
                tf.cast(tp, dtype=tf.float32),
                axis=-1),
            axis=-1)

        gt = tf.math.reduce_sum(
            tf.where(
                labels_true >= 0,
                1.0,
                0.0),
            axis=-1)

        recall = tf.math.divide_no_nan(tp, gt)
        recall = tf.cast(recall, dtype=self.dtype)

        n = tf.ones_like(recall)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, recall)
            recall = tf.multiply(recall, sample_weight)
            n = tf.multiply(n, sample_weight)

        self.sum_recall.assign_add(tf.reduce_sum(recall))
        self.num_samples.assign_add(tf.reduce_sum(n))

    def result(self):
        return tf.math.divide_no_nan(
            self.sum_recall,
            self.num_samples)
