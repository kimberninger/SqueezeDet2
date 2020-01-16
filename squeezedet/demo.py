import tensorflow as tf
from absl import app, flags

from squeezedet.models import detector
from squeezedet.layers import Fire
from squeezedet.utils import get_anchors, prepare_image, normalize_bboxes

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'mode',
    'image',
    '\'image\' or \'video\'.')
flags.DEFINE_string(
    'checkpoint',
    'models/squeezedet_trained.h5',
    'Path to the model parameter file.')
flags.DEFINE_string(
    'input_path',
    './data/sample.png',
    'Input image or video to be detected. Can process glob input.')
flags.DEFINE_string(
    'out_dir',
    './data/out/',
    'Directory to dump output image or video.')
flags.DEFINE_string(
    'demo_net',
    'squeezeDet',
    'Neural net architecture.')

image_width = 1248
image_height = 384

bgr_means = 103.939, 116.779, 123.68

colors = [(0., 191., 255., 1.)]


def load_image(filename):
    raw_image = tf.io.decode_image(
        tf.io.read_file(filename),
        expand_animations=False)

    image = prepare_image(raw_image, image_width, image_height, bgr_means)

    return {
        'image': image,
        'raw_image': raw_image,
        'name': tf.strings.split(filename, '/')[-1]
    }


def main(_):
    num_anchors_x, num_anchors_y = 78, 24
    anchor_shapes = [[36, 37], [366, 174], [115, 59],
                     [162, 87], [38, 90], [258, 173],
                     [224, 108], [78, 170], [72, 43]]
    anchor_boxes = get_anchors(anchor_shapes, num_anchors_x, num_anchors_y,
                               image_width, image_height)

    classes = ['car', 'pedestrian', 'cyclist']

    net = tf.keras.models.load_model(
        FLAGS.checkpoint,
        custom_objects={'Fire': Fire},
        compile=False)

    _, det = detector(
        net,
        image_width,
        image_height,
        3,
        len(classes),
        anchor_boxes,
        len(anchor_shapes))

    data = tf.data.Dataset.list_files(FLAGS.input_path) \
        .map(load_image).batch(1)

    for d in data:
        _, _, bboxes = det.predict(d)
        bboxes = normalize_bboxes(bboxes, image_width, image_height)

        output = tf.image.draw_bounding_boxes(
            [d['raw_image'][0]], [bboxes[0]], colors)
        tf.io.write_file(
            tf.strings.join([FLAGS.out_dir, d['name'][0]], '/'),
            tf.image.encode_png(tf.cast(output[0], dtype=tf.uint8)))


if __name__ == '__main__':
    app.run(main)
