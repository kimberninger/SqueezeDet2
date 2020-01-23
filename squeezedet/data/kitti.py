import tensorflow as tf
import tensorflow_datasets as tfds

from squeezedet.utils import denormalize_bboxes
from .utils import filter_classes, prepare_data


def kitti(anchor_boxes,
          classes=None,
          image_width=None,
          image_height=None,
          bgr_means=(103.939, 116.779, 123.68),
          split=tfds.Split.TRAIN,
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

    def transform(features):
        w = tf.cast(tf.shape(features['image'])[1], dtype=tf.float32)
        h = tf.cast(tf.shape(features['image'])[0], dtype=tf.float32)

        bboxes = denormalize_bboxes(features['objects']['bbox'], w, h,
                                    invert_y=True)

        return {
            'image': features['image'],
            'labels': features['objects']['type'],
            'bboxes': bboxes
        }

    all_classes = info.features['objects']['type'].names

    return prepare_data(
        filter_classes(data.map(transform), classes, all_classes),
        anchor_boxes, image_width, image_height, bgr_means)
