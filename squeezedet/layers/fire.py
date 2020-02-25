import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from functools import partial


class Fire(tfkl.Layer):
    def __init__(self, s1x1, e1x1, e3x3, weight_decay, stddev=0.01, **kwargs):
        super(Fire, self).__init__(**kwargs)
        self.s1x1 = s1x1
        self.e1x1 = e1x1
        self.e3x3 = e3x3
        self.weight_decay = weight_decay
        self.stddev = stddev

    def build(self, input_shape):
        conv = partial(
            tfkl.Conv2D,
            activation='relu',
            padding='same',
            kernel_initializer=tfk.initializers.TruncatedNormal(
                stddev=self.stddev),
            kernel_regularizer=tfk.regularizers.l2(self.weight_decay))

        self.sq1x1 = conv(
            name=f'{self.name}/squeeze1x1',
            filters=self.s1x1,
            kernel_size=1)

        self.ex1x1 = conv(
            name=f'{self.name}/expand1x1',
            filters=self.e1x1,
            kernel_size=1)

        self.ex3x3 = conv(
            name=f'{self.name}/expand3x3',
            filters=self.e3x3,
            kernel_size=3)

    def call(self, inputs):
        squeezed = self.sq1x1(inputs)
        return tf.concat([self.ex1x1(squeezed), self.ex3x3(squeezed)], 3,
                         name=f'{self.name}/concat')

    def get_config(self):
        config = super(Fire, self).get_config()
        config.update({
            's1x1': self.s1x1,
            'e1x1': self.e1x1,
            'e3x3': self.e3x3,
            'weight_decay': self.l2,
            'stddev': self.stddev
        })
        return config
