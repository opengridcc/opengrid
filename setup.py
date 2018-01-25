# -*- coding: utf-8 -*-

"""
A setuptools based setup module for opengrid.

Adapted from
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import os
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open

here = os.path.abspath(os.path.dirname(__file__))

# adapted from https://packaging.python.org/guides/single-sourcing-package-version/#single-sourcing-the-version
about = {}
with open(os.path.join(here, 'opengrid', '__about__.py')) as fp:
    exec (fp.read(), about)

# Get the long description from the README file
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#    long_description = f.read()

import subprocess

if subprocess.call(["pip", "install", "-r", os.path.join(here, "requirements.txt"), "-v", "--no-cache"]):
    raise Exception("Could not install dependencies")

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__summary__'],
    # long_description=long_description,
    url=about['__url__'],
    author=about['__author__'],
    author_email=about['__email__'],
    license=about['__license__'],

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],

    keywords=about['__keywords__'],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    # py_modules=["tmpo.py"],

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # Note: for creating the source distribution, they had to be included in the
    # MANIFEST.in as well.
    package_data={
        'opengrid': ['notebooks/*'],
    },
    include_package_data=True,
    data_files=[('', ['LICENSE', 'README.md', 'requirements.txt'])]
)
