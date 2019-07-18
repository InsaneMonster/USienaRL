USienaRL
*********

Luca Pasqualini - SAILab - University of Siena
############################################################

This is a *python 3.6* and above library for Reinforcement Learning (RL) experiments.

The idea behind this library is to generate an intuitive yet versatile system to generate RL agents, experiments, models, etc.
The library is modular, and allow for easy experiment iterations and validation. It is currently used in my research and it
was built first for that purpose.

**Features**

- Model, Agent, Environment, Experiment, Exploration Policy abstract classes with their interrelationships hidden inside the implementation
- Utility functions to run experiment in multiple iterations with automated folder setup and organization
- Many state-of-the-art algorithms already implemented in pre-defined models, including:
    - Tabular Temporal Difference Q-Learning, SARSA, Expected SARSA with
    - Deep Temporal Difference Q-Learning (DQN), SARSA, Expected SARSA
    - Double Deep Temporal Difference Q-Learning (DDQN)
    - Dueling Temporal Difference Q-Learning (DDDQN)
    - Vanilla Policy Gradient (VPG)
- Many state-o
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