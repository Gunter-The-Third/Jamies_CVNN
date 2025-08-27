from setuptools import setup, find_packages

setup(
    name='cvnn',
    version='0.1.0',
    description='A neural network framework supporting complex-valued neural networks',
    author='Jamie Keegan-Treloar',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    python_requires='>=3.7',
)
