import tensorflow as tf
import tensorflow.keras as tfk

from squeezedet.utils import normalize_bboxes


class ImagePlotter(tfk.callbacks.Callback):
    def __init__(self, detector, filename,
                 image_width, image_height, bgr_means,
                 box_color=[(255.0, 0.0, 0.0, 1.0)]):
        super(ImagePlotter, self).__init__()
        self.detector = detector

        self.image = tf.cast(tf.io.decode_image(
            tf.io.read_file(filename)), dtype=tf.float32)[tf.newaxis]

        self.input_image = tf.image.resize(
            self.image[..., ::-1] - bgr_means,
            (image_height, image_width))

        self.box_color = box_color

        self.doit = True

    def on_batch_begin(self, batch, logs=None):
        if self.doit:
            _, _, boxes = self.detector.predict(self.input_image)

            _, image_width, image_height, _ = tf.shape(self.image)

            boxes = tf.cast(boxes[0], dtype=tf.float32)
            boxes = normalize_bboxes(boxes, image_width, image_height)

            output = tf.image.draw_bounding_boxes(
                self.image, [boxes], self.box_color)

            with tf.summary.create_file_writer('logs').as_default():
                tf.summary.image('Detection example', output / 255, 0)

            self.doit = False

    def on_batch_end(self, batch, logs=None):
        if self.doit:
            _, _, boxes = self.detector.predict(self.input_image)

            _, image_width, image_height, _ = tf.shape(self.image)

            boxes = tf.cast(boxes[0], dtype=tf.float32)
            boxes = normalize_bboxes(boxes, image_width, image_height)

            output = tf.image.draw_bounding_boxes(
                self.image, [boxes], self.box_color)

            with tf.summary.create_file_writer('logs').as_default():
                tf.summary.image('Detection example', output / 255, 0)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass
