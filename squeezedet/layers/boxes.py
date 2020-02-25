import tensorflow as tf
import tensorflow.keras.layers as tfkl

from ..utils import get_anchors, safe_exp
from ..utils import bbox_to_center_size, bbox_to_min_max


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
    def __init__(self, **kwargs):
        super(ClipBoxes, self).__init__(**kwargs)

    def call(self, inputs):
        return bbox_to_center_size(
            tf.clip_by_value(bbox_to_min_max(inputs), 0., 1.))

    def get_config(self):
        config = super(ClipBoxes, self).get_config()
        return config


class BoxInterpretation(tfkl.Layer):
    def __init__(self, anchor_shapes, anchor_grid_height, anchor_grid_width,
                 exp_thresh=1.0, **kwargs):
        super(BoxInterpretation, self).__init__(**kwargs)
        self.anchor_shapes = anchor_shapes
        self.anchor_grid_height = anchor_grid_height
        self.anchor_grid_width = anchor_grid_width
        self.exp_thresh = exp_thresh

    def build(self, input_shape):
        self.clip = ClipBoxes()
        self.anchors = get_anchors(
            self.anchor_shapes,
            self.anchor_grid_height,
            self.anchor_grid_width)

    def call(self, inputs):
        labels, confidence, deltas = inputs

        det_boxes = self.clip(tf.concat([
            self.anchors[..., :2] + deltas[..., :2] * self.anchors[..., 2:],
            self.anchors[..., 2:] * safe_exp(deltas[..., 2:], self.exp_thresh)
        ], axis=-1))

        probs = labels * confidence[..., tf.newaxis]

        det_class = tf.math.argmax(probs, axis=-1)
        det_probs = tf.math.reduce_max(probs, axis=-1)

        return det_class, det_probs, det_boxes

    def get_config(self):
        config = super(BoxInterpretation, self).get_config()
        config.update({
            'anchor_shapes': self.anchor_shapes,
            'anchor_grid_width': self.anchor_grid_width,
            'anchor_grid_height': self.anchor_grid_height,
            'exp_thresh': self.exp_thresh
        })
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

        _, order = tf.math.top_k(det_probs, k=self.top_n_detection)

        boxes = tf.gather(det_boxes, order, batch_dims=1)

        boxes = bbox_to_min_max(boxes)

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

        boxes = bbox_to_center_size(result.nmsed_boxes)

        final_labels = tf.where(valid_mask, result.nmsed_classes, -1)
        final_probs = tf.where(valid_mask, result.nmsed_scores, 0.0)

        final_boxes = tf.where(
            tf.tile(valid_mask[..., tf.newaxis], multiples=(1, 1, 4)),
            boxes,
            0.0)

        return tf.cast(final_labels, dtype=tf.int32), final_probs, final_boxes

    def get_config(self):
        config = super(BoxFilter, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'top_n_detection': self.top_n_detection,
            'prob_thresh': self.prob_thresh,
            'nms_thresh': self.nms_thresh
        })
        return config
