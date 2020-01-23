import tensorflow as tf
import tensorflow_datasets as tfds

from squeezedet.utils import denormalize_bboxes
from .utils import filter_classes, prepare_data


def voc2007(anchor_boxes,
            classes=None,
            image_width=None,
            image_height=None,
            bgr_means=(103.939, 116.779, 123.68),
            split=tfds.Split.TRAIN,
            data_dir=None):
    return voc(anchor_boxes,
               '2007',
               classes,
               image_width,
               image_height,
               bgr_means,
               split,
               data_dir)


def voc2012(anchor_boxes,
            classes=None,
            image_width=None,
            image_height=None,
            bgr_means=(103.939, 116.779, 123.68),
            split=tfds.Split.TRAIN,
            data_dir=None):
    return voc(anchor_boxes,
               '2012',
               classes,
               image_width,
               image_height,
               bgr_means,
               split,
               data_dir)


def voc(anchor_boxes,
        year='2007',
        classes=None,
        image_width=None,
        image_height=None,
        bgr_means=(103.939, 116.779, 123.68),
        split=tfds.Split.TRAIN,
        data_dir=None):
    data, info = tfds.load(
        'voc/' + year,
        split=split,
        data_dir=data_dir,
        with_info=True)

    def transform(features):
        w = tf.cast(tf.shape(features['image'])[1], dtype=tf.float32)
        h = tf.cast(tf.shape(features['image'])[0], dtype=tf.float32)

        bboxes = denormalize_bboxes(features['objects']['bbox'], w, h)

        return {
            'image': features['image'],
            'labels': features['objects']['label'],
            'bboxes': bboxes
        }

    all_classes = info.features['objects']['label'].names

    return prepare_data(
        filter_classes(data.map(transform), classes, all_classes),
        anchor_boxes, image_width, image_height, bgr_means)
