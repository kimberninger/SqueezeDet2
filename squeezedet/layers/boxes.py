import tensorflow as tf
import tensorflow.keras.layers as tfkl

from squeezedet.utils import safe_exp, get_anchors
from squeezedet.utils import normalize_bboxes, denormalize_bboxes


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
        self.image_width = image_width
        self.image_height = image_height

    def call(self, inputs):
        return denormalize_bboxes(
            tf.clip_by_value(
                normalize_bboxes(inputs, self.image_width, self.image_height),
                0., 1.),
            self.image_width, self.image_height)

    def get_config(self):
        config = super(ClipBoxes, self).get_config()
        config.update({
            'image_width': self.image_width,
            'image_height': self.image_height
        })
        return config


class BoxInterpretation(tfkl.Layer):
    def __init__(self, anchor_shapes, anchor_grid_width, anchor_grid_height,
                 image_width, image_height,
                 exp_thresh=1.0, **kwargs):
        super(BoxInterpretation, self).__init__(**kwargs)
        self.anchor_shapes = anchor_shapes
        self.anchor_grid_width = anchor_grid_width
        self.anchor_grid_height = anchor_grid_height
        self.exp_thresh = exp_thresh
        self.image_width = image_width
        self.image_height = image_height

    def build(self, input_shape):
        self.clip = ClipBoxes(self.image_width, self.image_height)
        self.anchors = get_anchors(
            self.anchor_shapes,
            self.anchor_grid_width,
            self.anchor_grid_height,
            self.image_width,
            self.image_height)

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
        config.update({
            'anchor_shapes': self.anchor_shapes,
            'anchor_grid_width': self.anchor_grid_width,
            'anchor_grid_height': self.anchor_grid_height,
            'exp_thresh': self.exp_thresh,
            'image_width': self.image_width,
            'image_height': self.image_height
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

        order = tf.argsort(
            det_probs, direction='DESCENDING')[:, :self.top_n_detection+1]

        boxes = tf.gather(det_boxes, order, batch_dims=1)

        boxes = normalize_bboxes(boxes, 1.0, 1.0)

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

        boxes = denormalize_bboxes(result.nmsed_boxes, 1.0, 1.0)

        from absl import logging
        logging.set_verbosity(logging.ERROR)
        final_labels = tf.ragged.boolean_mask(result.nmsed_boxes, valid_mask)
        final_probs = tf.ragged.boolean_mask(result.nmsed_scores, valid_mask)
        final_boxes = tf.ragged.boolean_mask(boxes, valid_mask)
        logging.set_verbosity(logging.INFO)

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
