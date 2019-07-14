USienaRL
*********

Luca Pasqualini - SAILab - University of Siena
############################################################

TODO: CHANGE THE FOLLOWING README TO BE COMPLIANT TO USIENARL

This is a *python 3.6* and above implementation of the **NIST Test Suite for Random Number Generators** (RNGs).
The idea behind this work is to make a script oriented object-oriented framework for said tests.
This is born from my research since I required to use the tests inside a python research project and I found existing
implementation to be not well suited to that task without extensive modifications.

The NIST reference paper can be found at `SP800-22r1a <https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf>`_.

This work is inspired by the great work of David Johnston (C) 2017, which can be found on `github <https://github.com/dj-on-github/sp800_22_tests>`_.

**Features**

- All the test in the **NIST** paper vectorized and optimized the better I could
- Class structure for each test allowing for easy debug and use, both in script and inside broader applications
- Utility functions to pack the sequence in 8-bits using numpy and to run the tests in multiple ways
- Cache system both at function level and at test level to improve performance
- Built-in measurement of time required to perform each test
- Default Test class and Result class to allow eventual extension to additional tests

**License**

*BSD 3-Clause License*

For additional information check the provided license file.

**How to install**

If you only need to use the framework, just download the pip package *nistrng* and import the package in your scripts

If you want to improve/modify/extends the framework, or even just try my own simple benchmarks at home, download or clone
the git repository. You are welcome to open issues or participate in the project, especially if further optimization is achieved.

**How to use**

For a simple use case, refer to benchmark provided in the repository. For advanced use, refer to the built-in documentation
and to the provided source code in the repository.

**Current issues**

Currently the slow speed of both the Serial and Approximate Entropy tests is an open issue. Any solution or improvement is
welcome.