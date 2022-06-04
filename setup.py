import os
import setuptools

ROOT = os.path.dirname(__file__)

with open(os.path.join(ROOT,"requirements", "requirements.txt")) as f:
    required = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ruska",
    version="0.0.1",
    author="Philipp Jung",
    author_email="philippjung@posteo.de",
    maintainer_email='philippjung@posteo.de',
    description="Helper to carry out data-cleaning experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://localhost",
    packages=setuptools.find_packages(),
    data_files=[('requirements', ['requirements/requirements.txt'])],
    install_requires=required,
    license='Apache License 2.0',
    python_requires='>=3',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only'
    ],
)

