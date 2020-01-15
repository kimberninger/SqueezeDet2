import tensorflow as tf
from utils import bbox_transform_inv, batch_iou
from absl import logging


class KittiReader:
    def __init__(self,
                 class_names,
                 image_width,
                 image_height,
                 anchor_box,
                 exclude_hard_examples=False,
                 bgr_means=(103.939, 116.779, 123.68),
                 data_augmentation=True,
                 drift_x=150,
                 drift_y=100):
        self.classes = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                class_names,
                range(len(class_names)),
                key_dtype=tf.string,
                value_dtype=tf.int64),
            1)

        self.exclude_hard_examples = exclude_hard_examples
        self.bgr_means = bgr_means
        self.image_width = image_width
        self.image_height = image_height
        self.anchor_box = anchor_box
        self.data_augmentation = data_augmentation
        self.drift_x = drift_x
        self.drift_y = drift_y

    @staticmethod
    def _get_obj_level(obj):
        height = obj[6] - obj[4] + 1
        truncation = obj[0]
        occlusion = obj[1]
        if height >= 40 and truncation <= 0.15 and occlusion <= 0:
            return 1
        elif height >= 25 and truncation <= 0.3 and occlusion <= 1:
            return 2
        elif height >= 25 and truncation <= 0.5 and occlusion <= 2:
            return 3
        else:
            return 4

    @tf.function
    def process_path(self, file_path):
        label_file = tf.strings.regex_replace(file_path, '\\.png', '.txt')
        label_file = tf.strings.regex_replace(label_file, 'image_2', 'label_2')

        img = tf.io.decode_png(tf.io.read_file(file_path))

        img = tf.reverse(tf.cast(img, dtype=tf.float32), axis=[-1])
        img -= tf.constant([[self.bgr_means]])

        # TODO Add data augmentation.

        orig_h = tf.shape(img)[0]
        orig_w = tf.shape(img)[1]

        x_scale = tf.cast(self.image_width / orig_w, dtype=tf.float32)
        y_scale = tf.cast(self.image_height / orig_h, dtype=tf.float32)

        img = tf.image.resize(img, (self.image_width, self.image_height))

        lines = tf.strings.split(tf.io.read_file(label_file), '\n')
        obj = tf.strings.split(tf.strings.strip(lines), ' ') \
            .to_tensor(default_value='0')

        labels = self.classes.lookup(
            tf.strings.strip(tf.strings.lower(obj[:, 0])))

        bboxes = bbox_transform_inv(tf.strings.to_number(obj[:, 4:8]))

        # TODO Add filtering according to _get_obj_level.
        mask = tf.less(labels, self.classes.size()-1)

        labels = tf.cast(labels[mask], dtype=tf.int32)

        bboxes = tf.reshape(tf.reshape(bboxes[mask], (-1, 2, 2)) *
                            tf.stack([[[x_scale, y_scale]]]), (-1, 4))

        anchor_boxes = tf.cast(self.anchor_box, dtype=tf.float32)

        ious = batch_iou(bboxes, anchor_boxes)

        dists = tf.math.reduce_sum(tf.math.square(
            tf.expand_dims(bboxes, 1) - tf.expand_dims(anchors, 0)), -1)

        overlap_ids = tf.argsort(ious, direction='DESCENDING')
        dist_ids = tf.argsort(dists)

        # TODO Suppresses warning due to a bug in TensorFlow. This will be
        # fixed in the upcoming release
        logging.set_verbosity(logging.ERROR)
        candidates = tf.concat([
            tf.ragged.boolean_mask(
                overlap_ids,
                tf.sort(ious, direction='DESCENDING') > 0),
            dist_ids
            ], axis=1)
        logging.set_verbosity(logging.INFO)

        # TODO Eliminate the chance of duplicate IDs.
        anchor_ids = candidates.to_tensor()[:, 0]

        delta = tf.concat([
            (bboxes[:, :2] - tf.gather(anchor_boxes, anchor_ids)[:, :2]) /
            tf.gather(anchor_boxes, anchor_ids)[:, 2:],
            tf.math.log(bboxes[:, 2:] /
                        tf.gather(anchor_boxes, anchor_ids)[:, 2:])
        ], axis=1)

        return ({
            'image': img,
            'mask': tf.RaggedTensor.from_row_starts(anchor_ids, (0,))
        }, {
            'delta': tf.RaggedTensor.from_row_starts(delta, (0,)),
            'boxes': tf.RaggedTensor.from_row_starts(bboxes, (0,)),
            'labels': tf.ragged.map_flat_values(
                lambda label: tf.one_hot(label,
                                         tf.cast(self.classes.size() - 1,
                                                 dtype=tf.int32)),
                tf.RaggedTensor.from_row_starts(labels, (0,)))
        })
