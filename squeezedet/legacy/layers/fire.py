import tensorflow as tf
import tensorflow.keras.layers as tfkl


class Fire(tfkl.Layer):
    def __init__(self, s1x1, e1x1, e3x3, l2, stddev=0.01, **kwargs):
        super(Fire, self).__init__(**kwargs)
        self.s1x1 = s1x1
        self.e1x1 = e1x1
        self.e3x3 = e3x3
        self.l2 = l2
        self.stddev = stddev

    def build(self, input_shape):
        self.sq1x1 = tfkl.Conv2D(
            name=self.name+'/squeeze1x1',
            filters=self.s1x1,
            kernel_size=1,
            padding='same',
            activation='relu',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.stddev),
            kernel_regularizer=tf.keras.regularizers.l2(self.l2))

        self.ex1x1 = tfkl.Conv2D(
            name=self.name+'/expand1x1',
            filters=self.e1x1,
            kernel_size=1,
            padding='same',
            activation='relu',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.stddev),
            kernel_regularizer=tf.keras.regularizers.l2(self.l2))

        self.ex3x3 = tfkl.Conv2D(
            name=self.name+'/expand3x3',
            filters=self.e3x3,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.stddev),
            kernel_regularizer=tf.keras.regularizers.l2(self.l2))

    def call(self, inputs):
        squeezed = self.sq1x1(inputs)
        return tf.concat([self.ex1x1(squeezed), self.ex3x3(squeezed)], 3,
                         name=self.name+'/concat')

    def get_config(self):
        config = super(Fire, self).get_config()
        config.update({
            's1x1': self.s1x1,
            'e1x1': self.e1x1,
            'e3x3': self.e3x3,
            'l2': self.l2,
            'stddev': self.stddev
        })
        return config
