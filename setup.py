from setuptools import setup, find_packages

setup(
    name="face-recognition-app",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'opencv-python-headless',
        'numpy',
        'torch',
        'pillow',
        'facenet-pytorch'
    ],
    entry_points={
        'console_scripts': [
            'face-recognition=face_recognition_app:main',
        ],
    },
    author='Your Name',
    description='Face Recognition Application',
)