from setuptools import setup, find_packages

setup(
    name='my_yolov3_library',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pyyaml',
        # other dependencies
    ],
)
