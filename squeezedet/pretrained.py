import tensorflow.keras as tfk
from pkg_resources import resource_filename
from squeezedet.layers import Fire


def squeezedet_pretrained():
    filename = resource_filename('', 'models/squeezedet_pretrained.h5')
    return tfk.models.load_model(
        filename,
        custom_objects={'Fire': Fire},
        compile=False)
