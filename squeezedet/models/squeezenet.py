import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from functools import partial

from ..layers import Fire


def squeezenet(num_output, weight_decay=0.0001):
    fire = partial(Fire, weight_decay=weight_decay)
    pool = partial(tfkl.MaxPool2D, pool_size=3, strides=2, padding='same')

    conv = partial(
        tfkl.Conv2D, kernel_size=3, padding='same',
        kernel_initializer=tfk.initializers.TruncatedNormal(
            stddev=0.001),
        kernel_regularizer=tfk.regularizers.l2(weight_decay))

    return tfk.Sequential([
        conv(name='conv1', filters=64, strides=2, activation='relu',
             trainable=False),
        pool(name='pool1'),

        fire(name='fire2', s1x1=16, e1x1=64, e3x3=64),
        fire(name='fire3', s1x1=16, e1x1=64, e3x3=64),
        pool(name='pool3'),

        fire(name='fire4', s1x1=32, e1x1=128, e3x3=128),
        fire(name='fire5', s1x1=32, e1x1=128, e3x3=128),
        pool(name='pool5'),

        fire(name='fire6', s1x1=48, e1x1=192, e3x3=192),
        fire(name='fire7', s1x1=48, e1x1=192, e3x3=192),

        fire(name='fire8', s1x1=64, e1x1=256, e3x3=256),
        fire(name='fire9', s1x1=64, e1x1=256, e3x3=256),

        fire(name='fire10', s1x1=96, e1x1=384, e3x3=384),
        fire(name='fire11', s1x1=96, e1x1=384, e3x3=384),
        tfkl.Dropout(name='drop11', rate=0.5),

        conv(name='conv12', filters=num_output, strides=1)
    ])
