import setuptools

import cpu

install_requires = [line.strip() for line in open("requirements.txt").readlines()]

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="core-pytorch-utils",
    url="https://github.com/serend1p1ty/core-pytorch-utils",
    description="Core APIs for deep learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=cpu.__version__,
    author="serend1p1ty",
    author_email="zjli1997@163.com",
    license="MIT License",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    # By default, the source distribution only includes the minimal set of
    # python files without "requirements.txt". This will cause a "FileNotFoundError" when
    # installing from this source distribution. So we include these "requirements.txt"
    # by the "data_files" option.
    data_files=["requirements.txt", "tests/requirements.txt"],
)
