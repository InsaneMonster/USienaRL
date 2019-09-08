#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# USienaRL is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import setuptools

from setuptools import setup, find_packages

# Define function to read from README.rst file


def readme():
    with open("README.rst") as file:
        return file.read()


# Set all package values

name: str = "usienarl"
version: str = "0.4.2"
requirements: [] = ["numpy>=1.14.5", "scipy>=1.2.1", "pandas", "matplotlib"]
extras_require: {} = {"tensorflow": ["tensorflow>=1.8.0,<=1.13.1"], "tensorflow_gpu": ["tensorflow-gpu>=1.8.0,<=1.13.1"]}
packages: [] = find_packages()
url: str = "https://github.com/InsaneMonster/USienaRL"
lic: str = "BSD 3-Clause"
author: str = "Luca Pasqualini"
author_email: str = "pasqualini@diism.unisi.it"
description: str = "University of Siena Reinforcement Learning library - SAILab"
long_description: str = readme()
keywords: str = "Reinforcement Learning Reinforcement-Learning RL Machine Machine-Learning ML library framework SAILab USiena Siena"
include_package_data: bool = True
classifiers: [] = [
                    "Development Status :: 4 - Beta",
                    "License :: OSI Approved :: BSD License",
                    "Programming Language :: Python :: 3.6",
                    "Topic :: Scientific/Engineering :: Artificial Intelligence",
                  ]

# Run setup

setup(
    name=name,
    version=version,
    install_requires=requirements,
    extras_require=extras_require,
    packages=packages,
    url=url,
    license=lic,
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    keywords=keywords,
    include_package_data=include_package_data,
    classifiers=classifiers
)
