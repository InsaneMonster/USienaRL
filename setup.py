from setuptools import setup

name: str = "usienarl"
version: str = "0.1.3"
requirements: [] = ["tensorflow-gpu==1.8.0, tensorflow-gpu==1.9.0, tensorflow-gpu==1.10.0, tensorflow-gpu==1.10.1, tensorflow-gpu==1.11.0, tensorflow-gpu==1.12.0, tensorflow-gpu==1.12.2, tensorflow-gpu==1.13.1", "numpy>=1.14.5", "scipy>=1.2.1"]
packages: [] = ["usienarl"]
url: str = "https://github.com/InsaneMonster/USienaRL"
lic: str = "Creative Commons CC-BY 3.0"
author: str = "Luca Pasqualini"
author_email: str = "psqluca@gmail.com"
description: str = "University of Siena Reinforcement Learning library - SAILab"

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
)
