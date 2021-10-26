import setuptools

INSTALL_REQUIREMENTS = ["numpy", "opencv-python", "ffmpeg-python"]

setuptools.setup(
    name="cpu",
    url="https://github.com/serend1p1ty/core-pytorch-utils.git",
    description="Core APIs for deep learning.",
    version="1.0.0",
    author="serend1p1ty",
    author_email="zjli1997@163.com",
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIREMENTS,
)
