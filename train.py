import tensorflow as tf

import tensorflow_datasets as tfds

from absl import app, flags
from functools import partial

from squeezedet.models import squeezedet

from squeezedet.data import kitti, voc
from squeezedet.data import attach_anchors, padded_batch, resize_images


anchor_grid_width = 78
anchor_grid_height = 24

anchor_shapes = [
    [37./384, 36./1248],
    [174./384, 366./1248],
    [59./384, 115./1248],
    [87./384, 162./1248],
    [90./384, 38./1248],
    [173./384, 258./1248],
    [108./384, 224./1248],
    [170./384, 78./1248],
    [43./384, 72./1248]
]


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir',
    default='./checkpoints/',
    help='Where to save the training progress.')

flags.DEFINE_string(
    'data_dir',
    default=None,
    help='Where to save the training data.')

flags.DEFINE_integer(
    'batch_size',
    default=20,
    help='Batch size.')

flags.DEFINE_integer(
    'max_steps',
    default=5001,
    help='Number of training steps to run.')

flags.DEFINE_integer(
    'viz_steps',
    default=500,
    help='Frequency at which to save visualizations.')

flags.DEFINE_integer(
    'image_width',
    default=1248,
    help='The width the images should be scaled to.')

flags.DEFINE_integer(
    'image_height',
    default=384,
    help='The height the images should be scaled to.')

flags.DEFINE_list(
    'classes',
    default=['car', 'cyclist', 'pedestrian'],
    help='The classes used for bounding box classification.')

flags.DEFINE_enum(
    'dataset',
    default='kitti',
    enum_values=['kitti', 'voc2007', 'voc2012'],
    help='The dataset used for training.')


def main(_):
    datasets = {
        'kitti': partial(
            kitti, classes=FLAGS.classes, data_dir=FLAGS.data_dir),
        'voc2007': partial(
            voc, classes=FLAGS.classes, data_dir=FLAGS.data_dir, year='2007'),
        'voc2012': partial(
            voc, classes=FLAGS.classes, data_dir=FLAGS.data_dir, year='2012')
    }

    def get_data(split):
        data = datasets[FLAGS.dataset](split=split)
        data = resize_images(
            data,
            image_width=FLAGS.image_width,
            image_height=FLAGS.image_height)
        data = attach_anchors(
            data,
            anchor_shapes,
            anchor_grid_height,
            anchor_grid_width)
        return padded_batch(data, FLAGS.batch_size)

    params = FLAGS.flag_values_dict()

    params['anchor_shapes'] = anchor_shapes
    params['anchor_grid_width'] = anchor_grid_width
    params['anchor_grid_height'] = anchor_grid_height

    params['weight_decay'] = 0.0001

    estimator = tf.estimator.Estimator(
        squeezedet,
        params=params,
        config=tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=FLAGS.viz_steps),
        warm_start_from='squeezenet_trained')
    #   warm_start_from=tf.estimator.WarmStartSettings(
    #       ckpt_to_initialize_from='squeezenet_trained'))
    #       vars_to_warm_start=['.*']))

    train_input_fn = partial(get_data, split=tfds.Split.TRAIN)
    eval_input_fn = partial(get_data, split=tfds.Split.VALIDATION)

    for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
        estimator.train(train_input_fn, steps=FLAGS.viz_steps)
        eval_results = estimator.evaluate(eval_input_fn)
        print("Evaluation_results:\n\t%s\n" % eval_results)


if __name__ == "__main__":
    app.run(main)
