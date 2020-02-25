import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from squeezedet.layers import Fire


def squeezedet(num_classes, num_anchor_shapes, l2=0.0001, dropout_rate=0.5):
    num_output = num_anchor_shapes * (num_classes + 1 + 4)

    return tfk.Sequential([
        tfkl.Conv2D(
            name='conv1',
            filters=64,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            kernel_initializer=tfk.initializers.TruncatedNormal(stddev=0.001),
            kernel_regularizer=tfk.regularizers.l2(l2),
            trainable=False),
        tfkl.MaxPool2D(name='pool1', pool_size=3, strides=2, padding='same'),

        Fire(name='fire2', s1x1=16, e1x1=64, e3x3=64, l2=l2),
        Fire(name='fire3', s1x1=16, e1x1=64, e3x3=64, l2=l2),
        tfkl.MaxPool2D(name='pool3', pool_size=3, strides=2, padding='same'),

        Fire(name='fire4', s1x1=32, e1x1=128, e3x3=128, l2=l2),
        Fire(name='fire5', s1x1=32, e1x1=128, e3x3=128, l2=l2),
        tfkl.MaxPool2D(name='pool5', pool_size=3, strides=2, padding='same'),

        Fire(name='fire6', s1x1=48, e1x1=192, e3x3=192, l2=l2),
        Fire(name='fire7', s1x1=48, e1x1=192, e3x3=192, l2=l2),

        Fire(name='fire8', s1x1=64, e1x1=256, e3x3=256, l2=l2),
        Fire(name='fire9', s1x1=64, e1x1=256, e3x3=256, l2=l2),

        Fire(name='fire10', s1x1=96, e1x1=384, e3x3=384, l2=l2),
        Fire(name='fire11', s1x1=96, e1x1=384, e3x3=384, l2=l2),
        tfkl.Dropout(name='drop11', rate=dropout_rate),

        tfkl.Conv2D(
            name='conv12',
            filters=num_output,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=tfk.initializers.TruncatedNormal(stddev=0.0001),
            kernel_regularizer=tfk.regularizers.l2(l2))
    ])
