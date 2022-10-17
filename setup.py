from setuptools import setup, find_packages

setup(
    name='hyperevol',
    version='0.0.1',
    description='Algorithms for hyperparameter optimization',
    url='',
    author=['Laurits Tani'],
    author_email='laurits.tani@cern.ch',
    license='GPLv3',
    packages=find_packages(),
    package_data={
        'hyperevol': [
            'config/*',
            'tests/*',
            'scripts/*'
        ]
    },
    install_requires=[
        'docopt',
        'scipy',
        'six',
        'chaospy',
        'scikit-learn',
        'numpy',
        'xgboost',
        'scikit-optimize'
    ],
)
