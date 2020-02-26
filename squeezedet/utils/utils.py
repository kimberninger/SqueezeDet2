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
