#!/usr/bin/python
"""multichannelAnalysis setup script."""

from setuptools import setup

# Package Version
from adenine import __version__ as version

setup(
    name='multichannelAnalysis',
    version=version,

    description=('multichannelAnalysis'),
    long_description=open('README.md').read(),
    author="Vanessa D'Amario",
    author_email='vanessa.damario@dibris.unige.it',
    maintainer="Vanessa D'Amario",
    maintainer_email='vanessa.damario@dibris.unige.it',
    url='https://github.com/vanessadamario/multichannelAnalysis.git',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    license='FreeBSD',

    packages=['multichannelAnalysis'],
    install_requires=['numpy (>=1.10.1)',
                      'scipy (>=0.16.1)',
                      'scikit-learn (>=0.18)',
                      'matplotlib (>=1.5.1)',
                      'pywt'],
    scripts=['scripts/feat_extraction.py', 'scripts/prepare_data.py',
             'scripts/classification.py'],
)
