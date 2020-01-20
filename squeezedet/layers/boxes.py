import tensorflow as tf
import tensorflow.keras.layers as tfkl

from squeezedet.utils import (
    bbox_transform, bbox_transform_inv,
    normalize_bboxes, denormalize_bboxes, safe_exp)


class GatherAnchors(tfkl.Layer):
    def __init__(self, **kwargs):
        super(GatherAnchors, self).__init__(**kwargs)

    def call(self, inputs):
        x, ids = inputs
        return tf.gather(x, tf.math.maximum(0, ids), batch_dims=1)

    def compute_mask(self, inputs, mask=None):
        _, ids = inputs
        if mask is None:
            return ids >= 0
        return mask & (ids >= 0)

    def get_config(self):
        config = super(GatherAnchors, self).get_config()
        return config


class ClipBoxes(tfkl.Layer):
    def __init__(self, image_width, image_height, **kwargs):
        super(ClipBoxes, self).__init__(**kwargs)
        self.clip = tf.tile(tf.constant([
            image_width - 1.0,
            image_height - 1.0
        ]), (2,))

    def call(self, inputs):
        return bbox_transform_inv(
            tf.clip_by_value(bbox_transform(inputs), 0.0, self.clip))

    def get_config(self):
        config = super(ClipBoxes, self).get_config()
        return config


class BoxInterpretation(tfkl.Layer):
    def __init__(self, anchors,
                 image_width, image_height,
                 exp_thresh=1.0, **kwargs):
        super(BoxInterpretation, self).__init__(**kwargs)
        self.anchors = anchors
        self.exp_thresh = exp_thresh
        self.clip = ClipBoxes(image_width, image_height)

    def call(self, inputs):
        labels, confidence, deltas = inputs

        det_boxes = self.clip(tf.concat([
            self.anchors[:, :2] + deltas[:, :, :2] * self.anchors[:, 2:],
            self.anchors[:, 2:] * safe_exp(deltas[:, :, 2:], self.exp_thresh)
        ], axis=-1))

        probs = labels * confidence[..., tf.newaxis]

        det_class = tf.math.argmax(probs, axis=-1)
        det_probs = tf.math.reduce_max(probs, axis=-1)

        return det_class, det_probs, det_boxes

    def get_config(self):
        config = super(BoxInterpretation, self).get_config()
        return config


class BoxFilter(tfkl.Layer):
    def __init__(self, num_classes,
                 top_n_detection, prob_thresh, nms_thresh, **kwargs):
        super(BoxFilter, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.top_n_detection = top_n_detection
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh

    def call(self, inputs):
        det_class, det_probs, det_boxes = inputs

        order = tf.argsort(
            det_probs, direction='DESCENDING')[:, :self.top_n_detection+1]

        boxes = tf.gather(det_boxes, order, batch_dims=1)

        boxes = normalize_bboxes(boxes)

        scores = tf.gather(det_probs, order, batch_dims=1)
        onehot = tf.one_hot(
            tf.gather(det_class, order, batch_dims=1),
            depth=self.num_classes,
            on_value=True,
            off_value=False)

        boxes = tf.where(onehot[..., tf.newaxis], boxes[..., tf.newaxis, :], 0)
        scores = tf.where(onehot, scores[..., tf.newaxis], 0)

        result = tf.image.combined_non_max_suppression(
            boxes, scores,
            max_output_size_per_class=self.top_n_detection,
            max_total_size=self.top_n_detection,
            iou_threshold=self.nms_thresh,
            score_threshold=self.prob_thresh,
            clip_boxes=False,
            pad_per_class=True)

        valid = result.valid_detections[..., tf.newaxis]
        valid_mask = tf.range(self.top_n_detection)[tf.newaxis] < valid

        boxes = denormalize_bboxes(result.nmsed_boxes)

        final_labels = tf.ragged.boolean_mask(result.nmsed_boxes, valid_mask)
        final_probs = tf.ragged.boolean_mask(result.nmsed_scores, valid_mask)
        final_boxes = tf.ragged.boolean_mask(boxes, valid_mask)

        return tf.cast(final_labels, dtype=tf.int32), final_probs, final_boxes
