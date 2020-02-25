import tensorflow as tf
from absl import app

from squeezedet.models import squeezedet


def load_weights(model, checkpoint_file):
    import joblib
    weights = joblib.load(checkpoint_file)
    for layer in model.layers:
        if layer.name.startswith('conv'):
            if layer.name in weights:
                print(layer.name)
                kernel = tf.transpose(
                    weights[layer.name][0],
                    perm=(2, 3, 1, 0))
                bias = weights[layer.name][1]
                layer.set_weights((kernel, bias))
        if layer.name.startswith('fire'):
            for layer2 in [layer.sq1x1, layer.ex1x1, layer.ex3x3]:
                if layer2.name in weights:
                    print(layer2.name)
                    kernel = tf.transpose(
                        weights[layer2.name][0],
                        perm=(2, 3, 1, 0))
                    bias = weights[layer2.name][1]
                    layer2.set_weights((kernel, bias))


def main(_):
    model = squeezedet(3, 9)
    model.predict(tf.random.normal((20, 384, 1248, 3)))
    print(model.layers)

    load_weights(
        model,
        '/Users/kimberninger/Downloads/squeezeDet-master/model.ckpt-87000.pkl')
    model.save('squeezenet_trained')


if __name__ == '__main__':
    app.run(main)
