# -*- coding: utf-8 -*-

"""
A setuptools based setup module for opengrid.

Sources:
    - https://packaging.python.org/en/latest/distributing.html
    - https://github.com/pypa/sampleproject
    - https://packaging.python.org/guides/single-sourcing-package-version
    - See https://pypi.python.org/pypi?%3Aaction=list_classifiers
"""

from os import path
from setuptools import setup, find_packages

HERE = path.abspath(path.dirname(__file__))


with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='opengrid',
    version='0.6.0',
    description='Open-source algorithms for data-driven building analysis and control',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Roel De Coninck and others',
    maintainer="Evert Borghgraef",
    license='Apache 2.0',
    url='https://opengridcc.github.io',
    project_urls={},
    keywords='algorithms buildings monitoring analysis control',
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    package_dir={},
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'patsy',
        'statsmodels',
    ],
    include_package_data=True
)
