import tensorflow.keras as tfk
from pkg_resources import resource_filename
from squeezedet.layers import Fire


def load_model(resource_name):
    file_name = resource_filename(
        'squeezedet.models.pretrained',
        resource_name)
    return tfk.models.load_model(
        file_name,
        custom_objects={'Fire': Fire},
        compile=False)


def squeezedet_pretrained():
    return load_model('squeezedet_pretrained.h5')


def squeezedet_trained():
    return load_model('squeezedet_trained.h5')
