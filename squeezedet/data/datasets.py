import tensorflow as tf
import tensorflow_datasets as tfds

from .data_utils import filter_classes
from ..utils import bbox_to_center_size


def kitti(split=tfds.Split.TRAIN, classes=None, data_dir=None):
    def process_kitti(features):
        return {
            'image': tf.image.convert_image_dtype(
                features['image'], dtype=tf.float32)
        }, {
            'bboxes': bbox_to_center_size(
                features['objects']['bbox'], invert_y=True),
            'labels': tf.cast(features['objects']['type'], dtype=tf.int32)
        }

    data, info = tfds.load(
        'kitti',
        split=split,
        with_info=True,
        data_dir=data_dir,
        download_and_prepare_kwargs={
            'download_config': tfds.download.DownloadConfig(
                register_checksums=True)
        })
    return filter_classes(
        data.map(process_kitti),
        info.features['objects']['type'].names,
        classes)


def voc(year='2007', split=tfds.Split.TRAIN, classes=None, data_dir=None):
    def process_voc(features):
        return {
            'image': tf.image.convert_image_dtype(
                features['image'], dtype=tf.float32)
        }, {
            'bboxes': bbox_to_center_size(
                features['objects']['bbox'], invert_y=False),
            'labels': tf.cast(features['objects']['label'], dtype=tf.int32)
        }

    data, info = tfds.load(
        f'voc/{year}',
        split=split,
        with_info=True,
        data_dir=data_dir,
        download_and_prepare_kwargs={
            'download_config': tfds.download.DownloadConfig(
                register_checksums=True)
        })
    return filter_classes(
        data.map(process_voc),
        info.features['objects']['label'].names,
        classes)
