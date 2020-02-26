import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

import tensorflow_probability as tfp

from ..layers import BoxInterpretation, BoxFilter
from ..utils import iou, draw_bounding_boxes
from ..metrics import AveragePrecision, AverageRecall

from .squeezenet import squeezenet

tfd = tfp.distributions
tfb = tfp.bijectors


def squeezedet(features, labels, mode, params, config):
    weight_decay = params.get('weight_decay', 0.0001)

    coef_class = params.get('loss_coefficient_class', 1.0)

    coef_conf = params.get('loss_coefficient_confidence', 1.0)
    coef_conf_pos = params.get('loss_coefficient_confidence_positive', 3.75)
    coef_conf_neg = params.get('loss_coefficient_confidence_negative', 5.0)

    coef_bbox = params.get('loss_coefficient_confidence_bounding_boxes', 5.0)

    learning_rate = params.get('learning_rate', 0.01)
    momentum = params.get('momentum', 0.9)

    decay_steps = params.get('decay_steps', 10000)
    decay_factor = params.get('decay_factor', 0.5)

    max_grad_norm = params.get('max_grad_norm', 1.0)

    top_n_detection = params.get('top_n_detection', 10)
    prob_thresh = params.get('prob_thresh',  0.005)
    nms_thresh = params.get('nms_thresh', 0.4)

    iou_threshold = params.get('iou_threshold',  0.5)

    num_output = len(params['anchor_shapes']) * (
        len(params['classes']) + 1 + 4
    )

    output_shapes = len(params['anchor_shapes']) * tf.constant([
        len(params['classes']), 1, 4
    ])

    model = squeezenet(num_output, weight_decay)

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    predictions = model(
        255. * features['image'][..., ::-1] - [103.939, 116.779, 123.68],
        training=training)

    classes, confidence, deltas = tf.split(predictions, output_shapes, axis=-1)

    softmax = tfkl.Activation(tfk.activations.softmax)
    classes = softmax(tfkl.Reshape((-1, len(params['classes'])))(classes))

    sigmoid = tfkl.Activation(tfk.activations.sigmoid)
    confidence = sigmoid(tfkl.Reshape((-1,))(confidence))

    deltas = tfkl.Reshape((-1, 4))(deltas)

    deltas = tf.gather(deltas, [1, 0, 3, 2], axis=-1)

    det_class, det_probs, det_boxes = BoxInterpretation(
        params['anchor_shapes'],
        params['anchor_grid_height'],
        params['anchor_grid_width'])((classes, confidence, deltas))

    final_classes, final_probs, final_boxes = BoxFilter(
        num_classes=len(params['classes']),
        top_n_detection=top_n_detection,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh
    )((det_class, det_probs, det_boxes))

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'final_classes': final_classes,
                'final_probs': final_probs,
                'final_boxes': final_boxes
            },
            loss=None,
            train_op=None)

    tf.compat.v1.summary.image(
        'Image',
        draw_bounding_boxes(
            features['image'],
            final_boxes,
            final_classes,
            tf.eye(3)))

    mask = tf.cast(features['anchor_ids'] >= 0, dtype=tf.float32)
    num_objects = tf.math.reduce_sum(mask, axis=-1, keepdims=True)

    def gather_anchors(values):
        return tf.gather(
            values, tf.math.maximum(0, features['anchor_ids']), batch_dims=1)

    classes_loss = tf.math.reduce_sum(
        mask * tfk.losses.categorical_crossentropy(
            tf.one_hot(labels['labels'], depth=len(params['classes'])),
            gather_anchors(classes)) /
        num_objects,
        axis=-1)

    ious = iou(labels['bboxes'], gather_anchors(det_boxes))

    confidence_loss = tf.math.reduce_sum(
        mask * tf.math.square(ious - gather_anchors(confidence)) / num_objects,
        axis=-1)

    deltas_loss = tf.math.reduce_sum(
        mask * tf.math.reduce_mean(tf.math.square(
            labels['deltas'] - gather_anchors(deltas)), axis=-1) /
        num_objects,
        axis=-1)

    confidence_penalty = tf.math.reduce_sum(
        tf.math.square(confidence), axis=-1) - tf.math.reduce_sum(
            tf.math.square(
                tf.where(
                    features['anchor_ids'] >= 0,
                    gather_anchors(confidence), 0.0)),
            axis=-1)

    num_anchor_boxes = tf.cast(tf.math.reduce_prod([
        len(params['anchor_shapes']),
        params['anchor_grid_height'],
        params['anchor_grid_width']
    ]), dtype=tf.float32)

    confidence_penalty /= num_anchor_boxes - num_objects

    reg_losses = model.get_losses_for(None) + model.get_losses_for(features)

    total_loss = tf.math.add_n(reg_losses) + tf.math.reduce_mean(
        coef_class * classes_loss +
        coef_conf * (coef_conf_pos * confidence_loss +
                     coef_conf_neg * confidence_penalty) +
        coef_bbox * deltas_loss)

    train_op = None
    if training:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_factor,
                staircase=True),
            momentum=momentum,
            clipnorm=max_grad_norm)
        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

        update_ops = (
            model.get_updates_for(None) +
            model.get_updates_for(features)
        )

        minimize_op = optimizer.get_updates(
            total_loss,
            model.trainable_variables)[0]

        train_op = tf.group(minimize_op, *update_ops)

    precision = AveragePrecision(
        num_classes=len(params['classes']),
        iou_threshold=iou_threshold)

    recall = AverageRecall(
        num_classes=len(params['classes']),
        iou_threshold=iou_threshold)

    precision.update_state(
        labels['bboxes'], labels['labels'],
        final_boxes, final_classes, final_probs)

    recall.update_state(
        labels['bboxes'], labels['labels'],
        final_boxes, final_classes, final_probs)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'labels': final_classes,
            'bboxes': final_boxes,
            'probs': final_probs,
            'classes': classes,
            'deltas': deltas,
            'confidence': confidence
        },
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops={
            'precision': precision,
            'recall': recall
        })
