import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from squeezedet.layers import GatherAnchors, BoxInterpretation, BoxFilter
from squeezedet.losses import ClassLoss, ConfidenceLoss, BboxLoss


def detector(net,
             image_width,
             image_height,
             num_channels,
             num_classes,
             anchor_boxes,
             num_anchor_shapes,
             loss_coef_bbox=5.0,
             loss_coef_conf=1.0,
             loss_coef_conf_pos=75.0,
             loss_coef_conf_neg=100.0,
             loss_coef_class=1.0,
             learning_rate=0.01,
             decay_steps=10000,
             decay_factor=0.5,
             momentum=0.9,
             max_grad_norm=1.0,
             epsilon=1e-16,
             exp_thresh=1.0,
             nms_thresh=0.4,
             prob_thresh=0.005,
             top_n_detection=10):
    image_input = tfk.Input(name='image',
                            shape=(image_height, image_width, num_channels))
    anchor_ids = tfk.Input(name='anchor_ids', shape=(None,), dtype=tf.int32)

    num_output = num_anchor_shapes * tf.constant([num_classes, 1, 4])

    preds = net(image_input)

    labels, confidence, deltas = tf.split(preds, num_output, axis=-1)

    softmax = tfkl.Activation(tfk.activations.softmax)
    labels = softmax(tfkl.Reshape((-1, num_classes))(labels))

    sigmoid = tfkl.Activation(tfk.activations.sigmoid)
    confidence = sigmoid(tfkl.Reshape((-1,))(confidence))

    deltas = tfkl.Reshape((-1, 4))(deltas)

    det_class, det_probs, det_boxes = BoxInterpretation(
        anchor_boxes, image_width, image_height)((labels, confidence, deltas))

    bboxes = tf.concat([confidence[..., tf.newaxis], det_boxes], axis=-1)

    model = tfk.Model(
        inputs=[image_input, anchor_ids],
        outputs=[
            GatherAnchors(name='labels')((labels, anchor_ids)),
            GatherAnchors(name='bboxes')((bboxes, anchor_ids)),
            GatherAnchors(name='deltas')((deltas, anchor_ids))
        ])

    penalty = tf.math.reduce_sum(tf.math.square(confidence), axis=-1) - \
        tf.math.reduce_sum(tf.math.square(
            tf.where(anchor_ids >= 0,
                     tf.gather(confidence,
                               tf.maximum(0, anchor_ids),
                               batch_dims=1),
                     0.0)),
            axis=-1)

    normalizer = len(anchor_boxes) - tf.math.reduce_sum(
        tf.cast(anchor_ids >= 0, dtype=tf.float32), axis=-1)

    model.add_loss(
        loss_coef_conf * loss_coef_conf_neg *
        tf.math.reduce_mean(penalty / normalizer),
        inputs=True)

    optimizer = tfk.optimizers.SGD(
        learning_rate=tfk.optimizers.schedules.ExponentialDecay(
            learning_rate, decay_steps, decay_factor, staircase=True),
        clipnorm=max_grad_norm)

    model.compile(optimizer=optimizer, loss={
        'labels': ClassLoss(),
        'bboxes': ConfidenceLoss(epsilon=epsilon),
        'deltas': BboxLoss()
    }, loss_weights={
        'labels': loss_coef_class,
        'bboxes': loss_coef_conf * loss_coef_conf_pos,
        'deltas': loss_coef_bbox
    })

    final_classes, final_pros, final_boxes = BoxFilter(
        num_classes=num_classes,
        top_n_detection=top_n_detection,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh
    )((det_class, det_probs, det_boxes))

    det = tfk.Model(
        inputs=image_input,
        outputs=[final_classes, final_pros, final_boxes])

    return model, det
