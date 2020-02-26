import tensorflow as tf

from ..utils import get_anchors, iou


def attach_anchors(data, anchor_shapes, anchor_grid_height, anchor_grid_width):
    def transform(inputs, outputs):
        anchor_boxes = get_anchors(
            anchor_shapes, anchor_grid_height, anchor_grid_width)

        ious = iou(outputs['bboxes'][:, tf.newaxis], anchor_boxes[tf.newaxis])

        dists = tf.math.reduce_sum(tf.math.square(
            outputs['bboxes'][:, tf.newaxis] - anchor_boxes[tf.newaxis]),
            axis=-1)

        candidates = tf.argsort(tf.where(ious > 0, -ious, dists))

        anchor_ids = tf.zeros(tf.shape(candidates)[0], dtype=tf.int32)
        for i in range(tf.shape(candidates)[0]):
            available = tf.math.reduce_all(
                tf.math.not_equal(
                    candidates[i][..., tf.newaxis],
                    anchor_ids[:i][tf.newaxis]),
                axis=-1)

            next_index = candidates[i][available][0]

            anchor_ids = tf.tensor_scatter_nd_add(
                anchor_ids, [[i]], [next_index])

        anchors = tf.gather(anchor_boxes, anchor_ids)

        deltas = tf.concat([
            (outputs['bboxes'][:, :2] - anchors[:, :2]) / anchors[:, 2:],
            tf.math.log(outputs['bboxes'][:, 2:] / anchors[:, 2:])
        ], axis=1)

        inputs['anchor_ids'] = anchor_ids
        outputs['deltas'] = deltas

        return inputs, outputs

    return data.map(transform)


def resize_images(data, image_width=None, image_height=None):
    def transform(inputs, outputs):
        image = inputs['image']

        height = image_height or tf.shape(image)[0]
        width = image_width or tf.shape(image)[1]

        image = tf.image.resize(image, size=(height, width))

        inputs['image'] = image

        return inputs, outputs

    return data.map(transform)


def filter_classes(data, all_classes, classes=None):
    if classes is None:
        return data

    def transform(inputs, outputs):
        mapping = tf.math.equal(
            tf.reshape(tf.strings.lower(all_classes), shape=(1, 1, -1)),
            tf.reshape(tf.strings.lower(classes), shape=(1, -1, 1)))

        mask = tf.math.logical_and(
            tf.one_hot(
                outputs['labels'],
                depth=len(all_classes),
                off_value=False,
                on_value=True)[:, tf.newaxis, :],
            mapping)

        outputs['labels'] = tf.cast(tf.where(mask)[:, 1], dtype=tf.int32)
        outputs['bboxes'] = tf.boolean_mask(
            outputs['bboxes'],
            tf.math.reduce_any(tf.math.reduce_any(mask, axis=-1), axis=-1))

        return inputs, outputs

    return data.map(transform)


def padded_batch(data, batch_size):
    input_values = {'image': 0.}
    output_values = {'labels': -1, 'bboxes': 0.5}

    input_shapes = {'image': (None, None, 3)}
    output_shapes = {'labels': (None,), 'bboxes': (None, 4)}

    input_values['anchor_ids'] = -1
    output_values['deltas'] = 0.

    input_shapes['anchor_ids'] = (None,)
    output_shapes['deltas'] = (None, 4)

    return data.padded_batch(
        batch_size,
        padding_values=(input_values, output_values),
        padded_shapes=(input_shapes, output_shapes))
