from setuptools import setup

setup(name='mindscope_qc',
    version='0.1.0',
    packages=['mindscope_utilities'],
    include_package_data = True,
    description='Utilities for loading, and visualizing quality control data and metrics for the Allen Institute Mindscope program',
    url='https://github.com/AllenInstitute/mindscope_qc',
    author='Allen Institute',
    author_email='kater@alleninstitute.org, sean.mcculloch@alleninstitute.org,',
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