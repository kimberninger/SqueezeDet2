import tensorflow as tf

from squeezedet.utils import prepare_image, iou, get_anchors
from squeezedet.utils import normalize_bboxes, denormalize_bboxes


def filter_classes(data, classes, all_classes):
    all_classes = tf.strings.lower(all_classes)

    classes = all_classes if classes is None else tf.strings.lower(classes)

    m = all_classes[..., tf.newaxis] == classes
    relevant_labels = tf.math.reduce_any(m, axis=-1)
    label_indices = tf.math.reduce_sum(
        tf.cast(m, dtype=tf.int32) * tf.range(len(classes)), axis=-1)

    def transform(features):
        mask = tf.gather(relevant_labels, features['labels'])
        labels = tf.gather(label_indices, features['labels'])[mask]

        features.update({
            'labels': tf.one_hot(labels, depth=len(classes)),
            'bboxes': features['bboxes'][mask]
        })

        return features

    return data.map(transform)


def prepare_data(data,
                 anchor_shapes,
                 anchor_grid_width,
                 anchor_grid_height,
                 image_width,
                 image_height,
                 bgr_means=(103.939, 116.779, 123.68)):
    def transform(features):
        original_width = tf.shape(features['image'])[1]
        original_height = tf.shape(features['image'])[0]

        new_width = original_width if image_width is None else image_width
        new_height = original_height if image_height is None else image_height

        bboxes = denormalize_bboxes(
            normalize_bboxes(
                features['bboxes'], original_width, original_height),
            new_width, new_height, invert_y=True)

        image = prepare_image(
            features['image'], image_width, image_height, bgr_means)

        anchor_boxes = get_anchors(
            anchor_shapes,
            anchor_grid_width,
            anchor_grid_height,
            image_width,
            image_height)

        ious = iou(bboxes[:, tf.newaxis], anchor_boxes[tf.newaxis])

        dists = tf.math.reduce_sum(tf.math.square(
            bboxes[:, tf.newaxis] - anchor_boxes[tf.newaxis]), axis=-1)

        candidates = tf.argsort(tf.where(ious > 0, -ious, dists))

        anchor_ids = tf.zeros(tf.shape(candidates)[0], dtype=tf.int32)
        for i in range(tf.shape(candidates)[0]):
            available = tf.math.reduce_all(
                candidates[i][..., tf.newaxis] !=
                anchor_ids[:i][tf.newaxis], axis=-1)

            next_index = candidates[i][available][0]

            anchor_ids = tf.tensor_scatter_nd_add(
                anchor_ids, [[i]], [next_index])

        deltas = tf.concat([
            (bboxes[:, :2] - tf.gather(anchor_boxes, anchor_ids)[:, :2]) /
            tf.gather(anchor_boxes, anchor_ids)[:, 2:],
            tf.math.log(bboxes[:, 2:] /
                        tf.gather(anchor_boxes, anchor_ids)[:, 2:])
        ], axis=1)

        return {
            'image': image,
            'anchor_ids': anchor_ids
        }, {
            'labels': features['labels'],
            'bboxes': bboxes,
            'deltas': deltas
        }

    return data.map(transform)


def padded_batch(data, batch_size):
    return data.padded_batch(batch_size, padding_values=({
            'image': 0.,
            'anchor_ids': -1,
        }, {
            'labels': 0.,
            'bboxes': 0.,
            'deltas': 0.
        }), padded_shapes=({
            'image': [None, None, 3],
            'anchor_ids': [None]
        }, {
            'labels': [None, None],
            'bboxes': [None, 4],
            'deltas': [None, 4]
        }))
