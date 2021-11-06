from setuptools import find_packages, setup

setup(
    name='seizurecast',
    packages=find_packages(),
    install_requires=[
        'catch22==0.2.0',
        'matplotlib==3.4.2',
        'numpy==1.21.1',
        'pandas==1.3.2',
        'pyEDFlib==0.1.22',
        'scikit_learn==1.0.1',
        'scipy==1.7.0',
        'SQLAlchemy==1.4.26'
    ],
    version='0.1.1',
    description='ReReal-time forecasting epileptic seizure using electroencephalogram',
    author='Yanxian Lin',
    license='MIT',
)
