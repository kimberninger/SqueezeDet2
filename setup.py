from setuptools import setup, find_packages

setup(
    name='squeezedet',
    install_requires=[
        'tensorflow>=2.1.0',
        'tensorflow-datasets>=2.0.0'
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    package_data={
        'squeezedet.models.pretrained': ['*.h5'],
    }
)
