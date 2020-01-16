import tensorflow as tf


def normalize_bboxes(b,
                     image_width=1.0, image_height=1.0,
                     invert_y=False, switch_xy=False):
    w = tf.cast(image_width, dtype=tf.float32)
    h = tf.cast(image_height, dtype=tf.float32)
    return tf.stack([
        (b[..., 1] - b[..., 3] / 2) / h,
        (b[..., 0] - b[..., 2] / 2) / w,
        (b[..., 1] + b[..., 3] / 2) / h,
        (b[..., 0] + b[..., 2] / 2) / w
    ], -1)


def denormalize_bboxes(b,
                       image_width=1.0, image_height=1.0,
                       invert_y=False, switch_xy=False):
    w = tf.cast(image_width, dtype=tf.float32)
    h = tf.cast(image_height, dtype=tf.float32)
    return tf.stack([
        w * (b[..., 1] + b[..., 3]) / 2,
        h * ((b[..., 2] + b[..., 0]) / 2),
        w * (b[..., 3] - b[..., 1]),
        h * (b[..., 2] - b[..., 0])
    ], -1)


def bbox_transform(bbox):
    a = tf.constant([
        [1, 0, -0.5, 0],
        [0, 1, 0, -0.5],
        [1, 0, 0.5, 0],
        [0, 1, 0, 0.5]])

    b = tf.constant([0., 0, 1, 1])

    return tf.linalg.matvec(a, bbox) - b


def bbox_transform_inv(bbox):
    a = tf.constant([
        [0.5, 0, 0.5, 0],
        [0, 0.5, 0, 0.5],
        [-1, 0, 1, 0],
        [0, -1, 0, 1]])

    b = tf.constant([0.5, 0.5, 1, 1])

    return tf.linalg.matvec(a, bbox) + b


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


def prepare_image(image, width, height, bgr_means):
    """Prepares an image to be fed into the detector network."""
    im = tf.cast(image, dtype=tf.float32)[..., ::-1] - bgr_means
    return tf.image.resize(im, size=(height, width))


def draw_bboxes(images, bboxes):
    height, width = tf.shape(images)[1:3]
    bboxes = normalize_bboxes(bboxes, width, height)
    print(bboxes)
