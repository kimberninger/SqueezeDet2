import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from layers import Fire
from absl import app, flags

from models import squeezedet

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")


def load_weights(model, checkpoint_file):
    import joblib
    weights = joblib.load(checkpoint_file)
    for layer in model.layers:
        if layer.name.startswith('conv'):
            if layer.name in weights:
                print(layer.name)
                kernel = tf.transpose(weights[layer.name][0], (2, 3, 1, 0))
                bias = weights[layer.name][1]
                layer.set_weights((kernel, bias))
        if layer.name.startswith('fire'):
            for layer2 in [layer.sq1x1, layer.ex1x1, layer.ex3x3]:
                if layer2.name in weights:
                    print(layer2.name)
                    kernel = tf.transpose(weights[layer2.name][0], (2, 3, 1, 0))
                    bias = weights[layer2.name][1]
                    layer2.set_weights((kernel, bias))


def main(_):
    l2 = 0.0001
    model = tfk.Sequential([
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
        Fire(name='fire9', s1x1=64, e1x1=256, e3x3=256, l2=l2)
    ])

    model.predict(tf.random.normal((20, 384, 1248, 3)))

    load_weights(model, 'squeezenet_v1.1.pkl')
    model.save('models/squeezedet_pretrained.h5')

    model = squeezedet(3, 9)
    model.predict(tf.random.normal((20, 384, 1248, 3)))

    load_weights(model, 'model.ckpt-87000.pkl')
    model.save('models/squeezedet_trained.h5')

    model1 = tfk.models.load_model(
        'models/squeezedet_pretrained.h5', custom_objects={'Fire': Fire}, compile=False)
    model2 = tfk.models.load_model(
        'models/squeezedet_trained.h5', custom_objects={'Fire': Fire}, compile=False)

    x = tf.random.normal((1, 384, 1248, 3))
    print(model1.predict(x))
    print(model2.predict(x))


if __name__ == '__main__':
    app.run(main)
