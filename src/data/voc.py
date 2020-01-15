import tensorflow as tf
import tensorflow_datasets as tfds

from utils import iou
from absl import logging


def voc(classes,
        image_width,
        image_height,
        anchors,
        bgr_means=(103.939, 116.779, 123.68),
        data_augmentation=True,
        drift_x=150,
        drift_y=100,
        exclude_hard_examples=False,
        split=tfds.Split.TRAIN,
        year='2007',
        data_dir=None):
    data, info = tfds.load(
        'kitti',
        split=split,
        data_dir=data_dir,
        with_info=True,
        download_and_prepare_kwargs={
            'download_config': tfds.download.DownloadConfig(
                register_checksums=True)
        })

    bbox_transform = tf.constant([
        [0, image_width/2, 0, image_width/2],
        [-image_height/2, 0, -image_height/2, 0],
        [0, -image_width, 0, image_width],
        [-image_height, 0, image_height, 0]
    ])

    all_classes = tf.strings.lower(info.features['objects']['type'].names)
    m = tf.expand_dims(all_classes, -1) == tf.strings.lower(classes)
    relevant_labels = tf.math.reduce_any(m, -1)
    label_indices = tf.math.reduce_sum(
        tf.cast(m, dtype=tf.int32) * tf.range(len(classes)), -1)

    def transform(features):
        image = tf.cast(features['image'][..., ::-1], dtype=tf.float32)
        image -= tf.convert_to_tensor(bgr_means)

        # TODO Add data augmentation.

        orig_h = tf.cast(tf.shape(image)[0], dtype=tf.float32)
        orig_w = tf.cast(tf.shape(image)[1], dtype=tf.float32)

        image = tf.image.resize(image, size=(image_height, image_width))

        mask = tf.gather(relevant_labels, features['objects']['type'])

        # TODO Add filtering according to _get_obj_level.

        labels = tf.gather(label_indices, features['objects']['type'])[mask]

        bboxes = tf.linalg.matvec(
            bbox_transform,
            features['objects']['bbox'][mask] +
            tf.stack([-1 - 1/orig_h, 0, -1, 1/orig_w]))

        ious = iou(tf.expand_dims(bboxes, 1), tf.expand_dims(anchors, 0))

        dists = tf.math.reduce_sum(tf.math.square(
            tf.expand_dims(bboxes, 1) - tf.expand_dims(anchors, 0)), -1)

        overlap_ids = tf.argsort(ious, direction='DESCENDING')

        dist_ids = tf.argsort(dists)

        # TODO Suppresses warning due to a bug in TensorFlow. This will be
        # fixed in the upcoming release
        logging.set_verbosity(logging.ERROR)
        candidates = tf.ragged.boolean_mask(
            tf.concat([overlap_ids, dist_ids], 1),
            tf.concat([tf.sort(ious, direction='DESCENDING'),
                       tf.sort(dists, direction='DESCENDING')], 1) > 0)
        logging.set_verbosity(logging.INFO)

        candidates = candidates.to_tensor()

        anchor_ids = tf.zeros(tf.shape(candidates)[0], dtype=tf.int32)
        for i in range(tf.shape(candidates)[0]):
            available = tf.reduce_all(
                tf.expand_dims(candidates[i], -1) !=
                tf.expand_dims(anchor_ids[:i], 0), -1)
            next_index = candidates[i][available][0]
            anchor_ids = tf.tensor_scatter_nd_add(
                anchor_ids, [[i]], [next_index])

        deltas = tf.concat([
            (bboxes[:, :2] - tf.gather(anchors, anchor_ids)[:, :2]) /
            tf.gather(anchors, anchor_ids)[:, 2:],
            tf.math.log(bboxes[:, 2:] /
                        tf.gather(anchors, anchor_ids)[:, 2:])
        ], axis=1)

        return {
            'image': image,
            'anchor_ids': anchor_ids
        }, {
            'labels': tf.one_hot(labels, len(classes)),
            'bboxes': bboxes,
            'deltas': deltas
        }

    return data.map(transform)
