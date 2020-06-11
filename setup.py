#!/usr/bin/python
"""
A tool kit for analysis and visualization 
of Spatial Transcriptomics datasets
"""

import os
import io
import glob
from setuptools import setup, find_packages

# Get the long description from the relevant file
here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'stanalysis',
  version = "0.6.0",
  description = __doc__.split("\n", 1)[0],
  long_description = long_description,
  keywords = 'rna-seq analysis machine_learning spatial transcriptomics toolkit',
  author = 'Jose Fernandez Navarro',
  author_email = 'jc.fernandez.navarro@gmail.com',
  license = 'MIT',
  packages = find_packages(),
  include_package_data = False,
  package_data = {'': ['RELEASE-VERSION']},
  zip_safe = False,
  install_requires = [
    'setuptools',
    'argparse',
    'scipy',
    'numpy',
    'pandas',
    'sklearn',
    'matplotlib>=3.1.0',
    'torch',
    'umap-learn',
  ],
  #test_suite = 'tests',
  scripts = glob.glob('scripts/*.py'),
  classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'License :: MIT:: Copyright Jose Fernandez Navarro',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Environment :: Console',
  ],
)
