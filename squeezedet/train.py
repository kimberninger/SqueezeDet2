import tensorflow as tf
from absl import app, flags  # , logging

from squeezedet.data import kitti, padded_batch
from squeezedet.models import detector
from squeezedet.models.pretrained import squeezedet_pretrained

FLAGS = flags.FLAGS
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

    classes = ['car', 'pedestrian', 'cyclist']

    data = padded_batch(
        kitti(anchor_shapes, num_anchors_x, num_anchors_y,
              classes, image_width, image_height),
        FLAGS.batch_size)

    net = squeezedet_pretrained()

    model, _ = detector(
        net=net,
        image_width=image_width,
        image_height=image_height,
        num_channels=3,
        num_classes=len(classes),
        anchor_shapes=anchor_shapes,
        anchor_grid_width=num_anchors_x,
        anchor_grid_height=num_anchors_y)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.checkpoint_dir),
        tf.keras.callbacks.TensorBoard(log_dir=FLAGS.tensorboard_dir)
    ]

    model.fit(
        data,
        epochs=2,
        callbacks=callbacks,
        steps_per_epoch=7182//FLAGS.batch_size)


if __name__ == '__main__':
    app.run(main)
