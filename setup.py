#!/usr/bin/env python
import os
from setuptools import setup, find_packages

setup(
    name='torch-resize-images',
    version='0.1.2',
    description='A python library to resize images using PyTorch',
    url='https://github.com/NilsHendrikLukas/torch-resize-images',
    author='Nils Lukas',
    author_email='nlukas@uwaterloo.ca',
    scripts=["./torch_resize_new"],
    license='MIT',
    keywords='resize image pytorch',
    packages=find_packages('src'),
    install_requires=['Pillow>=5.1.0', 'torch', 'torchvision', 'gputil', 'tqdm']
)