from setuptools import setup

setup(name='hlm',
      version='1.2.0a1',
      description='Functions for fitting heteroskedastic linear (mixed) models to genetic data',
      url='http://github.com/alexTISYoung/hlm',
      author='Alexander I. Young',
      author_email='alextisyoung@gmail.com',
      license='MIT',
      scripts=['bin/hlmm.py'],
      classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Users',
            'Topic :: Statistical Genetics :: Analysis Tools',

            # Pick your license as you wish (should match "license" above)
            'License :: MIT License',

            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 2.7',
      ],
      keywords='statistics genetics heteroskedastic linear mixed model',
      packages=['hlm'],
      install_requires=[
            'numpy',
            'scipy',
            'argparse',
            'pysnptools'
        ],
      extras_require={
            'test': ['numdifftools'],
      },
      zip_safe=False)