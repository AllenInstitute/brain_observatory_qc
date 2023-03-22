from setuptools import setup

setup(name='brain_observatory_qc',
    version='0.1.0',
    packages=['brain_observatory_qc'],
    include_package_data = True,
    description='Utilities for loading, and visualizing quality control data and metrics for Allen Brain Observatory projects',
    url='https://github.com/AllenInstitute/brain_observatory_qc',
    author='Allen Institute',
    author_email='clark.roll@alleninstitute.org, sean.mcculloch@alleninstitute.org,',
    license='Allen Institute',
    install_requires=[
        'flake8',
        'pytest',
        'allensdk',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: Other/Proprietary License', # Allen Institute Software License
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8'
  ],
)