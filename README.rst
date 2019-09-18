USienaRL
*********

Luca Pasqualini - SAILab - University of Siena
############################################################

This is a *python 3.6* and above library for Reinforcement Learning (RL) experiments.

The idea behind this library is to generate an intuitive yet versatile system to generate RL agents, experiments, models, etc.
The library is modular, and allow for easy experiment iterations and validation. It is currently used in my research and it
was built first for that purpose.

**Note:** this is not meant to be an entry level framework. Users are expected to be at least somewhat experienced in both
python and reinforcement learning. The framework is easy to extend but almost all the agent-related and environment-related
work should be done by yourself.

**Features**

*Included in package:*

- Model, Agent, Environment, Experiment, Exploration Policy abstract classes with their interrelationships hidden inside the implementation
- Customizable experiments in which, besides the default metrics, additional metrics can be used to validate or pass the experiment:
    - Metrics of experiments are always related to rewards obtained (per-step or per-episode) and training episodes (usually the minimum the better)
- Utility functions to run the same experiment in multiple equal iterations with automated folder setup and organization, registering the following metrics:
    - Average total reward (reward in one episode) over training, validation and test
    - Average scaled reward (reward per step) over training, validation and test
    - Mean and standard deviation of average total and scaled reward over test in all experiment iterations (if more than one)
    - Mean and standard deviation of maximum total and scaled reward over test in all experiment iterations (if more than one)
    - Mean and standard deviation of minimum training episodes over test in all experiment iterations (if more than one)
    - Experiment iteration achieving best results in each one of the metrics described above
    - Easy to use .csv file with all the results for each experiment iteration
    - Plots of total and scaled rewards over both all training and validation episodes, saved as .png files
- Many state-of-the-art algorithms already implemented in pre-defined models, including:
    - Tabular Temporal Difference Q-Learning, SARSA, Expected SARSA with Prioritized Experience Replay memory buffer
    - Deep Temporal Difference Q-Learning (DQN), SARSA, Expected SARSA with Prioritized Experience Replay memory buffer
    - Double Deep Temporal Difference Q-Learning (DDQN) with Prioritized Experience Replay memory buffer
    - Dueling Temporal Difference Q-Learning (DDDQN) with Prioritized Experience Replay memory buffer
    - Vanilla Policy Gradient (VPG) with General Advantage Estimate buffer using rewards-to-go
- Many state-of-the-art exploration policies, including:
    - Epsilon Greedy with tunable decay rate, start value and end value
    - Boltzmann sampling with tunable temperature decay rate, start value and end value
- Config class to define the hidden layers of all Tensorflow graphs (including the CNN)
- Default Pass-Through interface class to allow communications between agents and environments
- Additive action mask for all the algorithms supporting it (only discrete action sets):
    - The mask supports two values: -infinity (mask) and 0.0 (pass-through)
    - If not supplied, the mask is by default full pass-through

*Not included in package:*

- Extensive set of benchmarks for each algorithm in the OpenAI gym environment
- Default agents, OpenAI gym environment and benchmark experiment classes to test the benchmarks by yourself

For additional example of usage of this framework, take a look a these github pages:

- `TicTacToeRL <https://github.com/InsaneMonster/TicTacToeRL>`_
- `UltimateTicTacToeRL <https://github.com/InsaneMonster/UltimateTicTacToeRL>`_

**License**

*BSD 3-Clause License*

For additional information check the provided license file.

**How to install**

If you only need to use the framework, just download the pip package *usienarl* and import the package in your scripts.

When installing, make sure to choose the version suiting your computing capabilities.
If you have CUDA installed, the gpu version is advised. Otherwise, just use the cpu version.
To choose a version, specify your extra require during install:

- pip install usienarl[tensorflow-gpu] to install the tensorflow-gpu version
- pip install usienarl[tensorflow] to install the tensorflow using cpu version

**Note:** failure in specifying the extra require will cause tensorflow to not be installed, and as such the library won't
be usable at all. For instance, this is not allowed, *unless you already have tensorflow installed*:

- pip install usienarl

If you want to improve/modify/extends the framework, or even just try my own benchmarks at home, download or clone
the `git repository <https://github.com/InsaneMonster/USienaRL>`_.
You are welcome to open issues or participate in the project. Note that the benchmarks are built to run using tensorflow-gpu.

**Requirements**

Besides Tensorflow, with this package also the following packages will be installed in your environment:

- NumPy
- SciPy
- Pandas
- Matplotlib

**How to use**

For a simple use case, refer to benchmark provided in the `repository <https://github.com/InsaneMonster/USienaRL>`_. For advanced use, refer to the built-in documentation
and to the provided source code in the `repository <https://github.com/InsaneMonster/USienaRL>`_.

**Current issues**

From the save-restore standpoint it could be useful to implement an easy way to pass a metagraph
without the need to redefine the entire agent (maybe serializing the agent somehow?).

An experiment can right now work only in a specific environment. It could be interesting to test multiple environments
both from a curriculum learning perspective (it can still be done using multiple subsequent experiments, however) and from
a generalization perspective (train one one, validate on another, etc).

A way to check if environments are compatible one another would be required too if what said above is implemented.

**Changelog**

- Removed Trust Region Policy Optimization algorithm (under development) from main library until it is completed.
- Fixed wrong parameter in display cycle metrics method in experiment class