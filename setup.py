from setuptools import setup

README_FILE: str = "README.rst"
REQUIREMENTS_FILE: str = "requirements.txt"

name: str = "usienarl"
version: str = "0.1.1"

with open(REQUIREMENTS_FILE) as file:
    requirements = file.read().splitlines()

packages: [] = ['usienarl']
url: str = "https://github.com/InsaneMonster/USienaRL"
lic: str = "Creative Commons CC-BY 3.0"
author: str = "Luca Pasqualini"
author_email: str = "psqluca@gmail.com"
description: str = "University of Siena Reinforcement Learning library - SAILab"

with open(README_FILE) as file:
    long_description: str = file.read()

setup(
    name=name,
    version=version,
    install_requires=requirements,
    packages=packages,
    url=url,
    license=lic,
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description
)
