from setuptools import setup, find_packages

setup(
    name='squeezedet',
    install_requires=[
        'tensorflow>=2.1.0',
        'tensorflow-datasets>=1.3.2'
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    package_data={
        'squeezedet.pretrained': ['*.h5'],
    }
)
