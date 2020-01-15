import tensorflow as tf
from absl import app, flags  # , logging

from data import kitti
from models import squeezeDet
from callbacks import ImagePlotter
from utils import get_anchors

FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 1000000,
                     'Number of epochs to train.')
flags.DEFINE_string('tensorboard_dir', 'logs',
                    'Where to write the TensorBoard logs.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                    'Where to write the model checkpoints.')
flags.DEFINE_string('data_path', 'KITTI/training',
                    'Where the data is located.')
flags.DEFINE_integer('batch_size', '20',
                     'The batch size to apply to the data.')

flags.DEFINE_string(
    'dataset',
    'KITTI',
    'Currently only support KITTI dataset.')
flags.DEFINE_string(
    'data_path',
    '',
    'Root directory of data.')
flags.DEFINE_string(
    'image_set',
    'train',
    'Can be \'train\', \'trainval\', \'val\', or \'test\'.')
flags.DEFINE_string(
    'year',
    '2007',
    'VOC challenge year. 2007 or 2012. Only used for Pascal VOC dataset')
flags.DEFINE_string(
    'train_dir',
    '/tmp/bichen/logs/squeezeDet/train',
    'Directory where to write event logs and checkpoint.')
flags.DEFINE_integer(
    'max_steps',
    1000000,
    'Maximum number of batches to run.')
flags.DEFINE_string(
    'net',
    'squeezeDet',
    'Neural net architecture.')
flags.DEFINE_string(
    'pretrained_model_path',
    '',
    'Path to the pretrained model.')
flags.DEFINE_integer(
    'summary_step',
    10,
    'Number of steps to save summary.')
flags.DEFINE_integer(
    'checkpoint_step',
    1000,
    'Number of steps to save summary.')


def main(_):
    image_width = 1248
    image_height = 384

    num_anchors_x, num_anchors_y = 78, 24
    anchor_shapes = [[36, 37], [366, 174], [115, 59],
                     [162, 87], [38, 90], [258, 173],
                     [224, 108], [78, 170], [72, 43]]
    anchor_boxes = get_anchors(anchor_shapes, num_anchors_x, num_anchors_y,
                               image_width, image_height)

    classes = ['car', 'pedestrian', 'cyclist']

    data = kitti(classes, image_width, image_height, anchor_boxes) \
        .padded_batch(FLAGS.batch_size, padding_values=({
            'image': 0.,
            'anchor_ids': -1,
        }, {
            'labels': 0.,
            'bboxes': 0.,
            'deltas': 0.
        }), padded_shapes=({
            'image': [image_height, image_width, 3],
            'anchor_ids': [None]
        }, {
            'labels': [None, len(classes)],
            'bboxes': [None, 4],
            'deltas': [None, 4]
        }))

    model, detector = squeezeDet(
        image_width=image_width,
        image_height=image_height,
        num_channels=3,
        num_classes=len(classes),
        anchor_boxes=anchor_boxes,
        num_anchor_shapes=len(anchor_shapes),
        loss_coef_bbox=1.0,
        loss_coef_conf=1.0,
        loss_coef_conf_pos=1.0,
        loss_coef_conf_neg=1.0,
        loss_coef_class=1.0)

    callbacks = [
        # tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.checkpoint_dir),
        tf.keras.callbacks.TensorBoard(log_dir=FLAGS.tensorboard_dir),
        ImagePlotter(detector, '/Users/kimberninger/TensorFlowTest/sample.png',
                     image_width, image_height, (103.939, 116.779, 123.68))
    ]

    for inputs, outputs in data:
        print(inputs['anchor_ids'])

    model.fit(
        data,
        epochs=2,
        callbacks=callbacks,
        steps_per_epoch=7182//FLAGS.batch_size)


if __name__ == '__main__':
    app.run(main)
