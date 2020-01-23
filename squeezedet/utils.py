import tensorflow as tf


def normalize_bboxes(b, image_width, image_height,
                     invert_x=False, invert_y=False, switch_xy=True):
    w = tf.cast(image_width, dtype=tf.float32)
    h = tf.cast(image_height, dtype=tf.float32)
    i = 1 if invert_x else 0
    j = 1 if invert_y else 0
    k = 1 if switch_xy else 0
    if switch_xy:
        w, h, i, j = h, w, j, i
    return tf.stack([
        i + ((1-2*i) * b[..., 0+k] - b[..., 2+k] / 2) / w,
        j + ((1-2*j) * b[..., 1-k] - b[..., 3-k] / 2) / h,
        i + ((1-2*i) * b[..., 0+k] + b[..., 2+k] / 2) / w,
        j + ((1-2*j) * b[..., 1-k] + b[..., 3-k] / 2) / h
    ], axis=-1)


def denormalize_bboxes(b, image_width, image_height,
                       invert_x=False, invert_y=False, switch_xy=True):
    w = tf.cast(image_width, dtype=tf.float32)
    h = tf.cast(image_height, dtype=tf.float32)
    i = 1 if invert_x else 0
    j = 1 if invert_y else 0
    k = 1 if switch_xy else 0
    if switch_xy:
        i, j = j, i
    return tf.stack([
        w * (j + (1-2*j) * (b[..., 2+k] + b[..., 0+k]) / 2),
        h * (i + (1-2*i) * (b[..., 3-k] + b[..., 1-k]) / 2),
        w * (b[..., 2+k] - b[..., 0+k]),
        h * (b[..., 3-k] - b[..., 1-k])
    ], axis=-1)


def iou(b1, b2, epsilon=0.0):
    xmin = tf.maximum(b1[..., 0] - b1[..., 2]/2, b2[..., 0] - b2[..., 2]/2)
    xmax = tf.minimum(b1[..., 0] + b1[..., 2]/2, b2[..., 0] + b2[..., 2]/2)
    ymin = tf.maximum(b1[..., 1] - b1[..., 3]/2, b2[..., 1] - b2[..., 3]/2)
    ymax = tf.minimum(b1[..., 1] + b1[..., 3]/2, b2[..., 1] + b2[..., 3]/2)
    intersection = tf.maximum(0.0, xmax - xmin) * tf.maximum(0.0, ymax - ymin)
    union = b1[..., 2] * b1[..., 3] + b2[..., 2] * b2[..., 3] - intersection
    return intersection / (union + epsilon)


def get_anchors(shapes, num_x, num_y, width, height):
    anchor_shapes = tf.convert_to_tensor(shapes, dtype=tf.float32)
    center_x = width * tf.linspace(0., 1., num_x+2)[1:-1]
    center_y = height * tf.linspace(0., 1., num_y+2)[1:-1]
    x, y, width = tf.meshgrid(center_x, center_y, anchor_shapes[:, 0])
    _, _, height = tf.meshgrid(center_x, center_y, anchor_shapes[:, 1])
    return tf.reshape(tf.stack([x, y, width, height], axis=3), shape=(-1, 4))


def safe_exp(w, t):
    return tf.where(
        w > t,
        tf.math.exp(tf.cast(t, dtype=tf.float32)) * (w - t + 1),
        tf.math.exp(w))


def prepare_image(image,
                  width=None,
                  height=None,
                  means=None,
                  reverse=True):
    """Prepares an image to be fed into the detector network."""
    im = tf.cast(image, dtype=tf.float32)

    if reverse:
        im = im[..., ::-1]

    if means is not None:
        im -= means

    w = tf.shape(im)[-2] if width is None else width
    h = tf.shape(im)[-3] if height is None else height

    if width is None and height is None:
        return im

    return tf.image.resize(im, size=(h, w))
