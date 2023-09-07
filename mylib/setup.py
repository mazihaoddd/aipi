from setuptools import setup, find_packages

setup(
    name='my_yolo_library',
    version='1.0',
    author="mazihao",
    author_email="1219384284@qq.com",
    description="",
    long_description="yololib",
    long_description_content_type="yololib",
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'torchvision>=0.9.0',
        'numpy>=1.18.5',
        'opencv-python>=4.1.2',
        'PyYAML>=5.3.1',
        # other dependencies
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

