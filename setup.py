from setuptools import find_packages, setup

setup(
    name='seizurecast',
    packages=find_packages(),
    install_requires=[
        'pyeeg @ git+https://github.com/forrestbao/pyeeg@e5d34f8e8dfd976b3c52e6c58f80306028275798',
        'catch22==0.2.0',
        'matplotlib==3.4.2',
        'numpy==1.21.1',
        'pandas==1.3.2',
        'pyEDFlib==0.1.22',
        'scikit_learn==1.0.1',
        'scipy==1.7.0',
        'SQLAlchemy==1.4.26'
    ],
    version='0.1.0',
    description='ReReal-time forecasting epileptic seizure using electroencephalogram',
    author='Yanxian Lin',
    license='MIT',
)
