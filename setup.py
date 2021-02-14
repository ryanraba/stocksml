from setuptools import setup, find_packages

with open('README.md', "r") as fid:   #encoding='utf-8'
    long_description = fid.read()

setup(
    name='stocksml',
    version='0.0.1',
    description='Stock Prediction using Machine Learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Ryan Raba',
    author_email='stocksml@ryanraba.com',
    url='https://github.com/ryanraba/stocksml',
    license='Apache-2.0',
    packages=find_packages(),
    install_requires=['bokeh>=1.4.0',
                      'dask>=2.13.0',
                      'distributed>=2.9.3',
                      'graphviz>=0.13.2',
                      'matplotlib>=3.1.2',
                      'numba==0.48.0',
                      'numcodecs>=0.6.3',
                      'numpy>=1.18.1',
                      'pandas>=0.25.2',
                      'scipy>=1.4.1',
                      'scikit-learn>=0.22.2',
                      'toolz>=0.10.0',
                      'xarray>=0.15.0',
                      'zarr>=2.3.2',
                      'fsspec>=0.6.2',
                      'pandas - datareader==0.9.0',
                      'keras==2.4.3',
                      'tensorflow==2.4.1'
                      ]
)
