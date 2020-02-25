import tensorflow as tf


def iou(b1, b2, epsilon=0.0):
    ymin = tf.maximum(b1[..., 0] - b1[..., 2]/2, b2[..., 0] - b2[..., 2]/2)
    ymax = tf.minimum(b1[..., 0] + b1[..., 2]/2, b2[..., 0] + b2[..., 2]/2)
    xmin = tf.maximum(b1[..., 1] - b1[..., 3]/2, b2[..., 1] - b2[..., 3]/2)
    xmax = tf.minimum(b1[..., 1] + b1[..., 3]/2, b2[..., 1] + b2[..., 3]/2)
    intersection = tf.maximum(0.0, xmax - xmin) * tf.maximum(0.0, ymax - ymin)
    union = b1[..., 2] * b1[..., 3] + b2[..., 2] * b2[..., 3] - intersection
    return intersection / (union + epsilon)


def get_anchors(shapes, anchor_grid_height, anchor_grid_width):
    anchor_shapes = tf.convert_to_tensor(shapes, dtype=tf.float32)
    center_y = tf.linspace(0., 1., 2*anchor_grid_height+1)[1::2]
    center_x = tf.linspace(0., 1., 2*anchor_grid_width+1)[1::2]
    x, y, width = tf.meshgrid(center_x, center_y, anchor_shapes[:, 1])
    _, _, height = tf.meshgrid(center_x, center_y, anchor_shapes[:, 0])
    return tf.reshape(tf.stack([y, x, height, width], axis=3), shape=(-1, 4))


def bbox_to_min_max(bboxes, invert_y=False):
    invert_y = 1. if invert_y else 0.
    return tf.stack([
        invert_y + ((1 - 2 * invert_y) * bboxes[..., 0] - bboxes[..., 2] / 2),
        bboxes[..., 1] - bboxes[..., 3] / 2,
        invert_y + ((1 - 2 * invert_y) * bboxes[..., 0] + bboxes[..., 2] / 2),
        bboxes[..., 1] + bboxes[..., 3] / 2
    ], axis=-1)


def bbox_to_center_size(bboxes, invert_y=False):
    return tf.stack([
        invert_y + (1 - 2 * invert_y) * (bboxes[..., 2] + bboxes[..., 0]) / 2,
        (bboxes[..., 3] + bboxes[..., 1]) / 2,
        bboxes[..., 2] - bboxes[..., 0],
        bboxes[..., 3] - bboxes[..., 1]
    ], axis=-1)


def draw_bounding_boxes(images, boxes, labels, colors):
    def draw(inputs):
        image, boxes, labels = inputs
        return tf.image.draw_bounding_boxes(
            image[tf.newaxis],
            boxes[labels >= 0][tf.newaxis],
            tf.gather(colors, labels[labels >= 0]))[0]
    return tf.map_fn(draw,
                     (images, bbox_to_min_max(boxes), labels),
                     dtype=tf.float32)


def safe_exp(w, t):
    return tf.where(
        w > t,
        tf.math.exp(tf.cast(t, dtype=tf.float32)) * (w - t + 1),
        tf.math.exp(w))


def true_and_false_positives(bboxes_true, labels_true,
                             bboxes_pred, labels_pred, confidence,
                             num_classes, iou_threshold=0.5):
    order = tf.argsort(confidence, direction='DESCENDING')

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


def mean_average_precision(bboxes_true, labels_true,
                           bboxes_pred, labels_pred, confidence,
                           num_classes, iou_threshold=0.5,
                           num_thresholds=11):
    thresholds = tf.linspace(0., 1., num_thresholds)

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

    return tf.math.reduce_mean(
        tf.math.reduce_max(
            precision[..., tf.newaxis] * tf.cast(
                tf.math.greater_equal(
                    recall[..., tf.newaxis],
                    thresholds[tf.newaxis, tf.newaxis]), dtype=tf.float32),
            axis=0))
