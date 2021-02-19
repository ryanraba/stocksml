from setuptools import setup, find_packages

with open('README.md', "r") as fid:   #encoding='utf-8'
    long_description = fid.read()

setup(
    name='stocksml',
    version='0.0.2',
    description='Stock Prediction using Machine Learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Ryan Raba',
    author_email='stocksml@ryanraba.com',
    url='https://github.com/ryanraba/stocksml',
    license='GPL-3.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['matplotlib>=3.1.2',
                      'numpy>=1.18.1',
                      'pandas>=0.25.2',
                      'scipy>=1.4.1',
                      'scikit-learn>=0.22.2',
                      'pandas-datareader>=0.9.0',
                      'keras>=2.4.3',
                      'tensorflow>=2.4.1']
)
