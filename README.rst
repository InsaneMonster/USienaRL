USienaRL
*********

This is a *python 3.6* and above library for Reinforcement Learning (RL) experiments.

The idea behind this library is to generate an intuitive yet versatile system to generate RL agents, experiments, models, etc.
The library is modular, and allow for easy experiment iterations and validation. It is currently used in my research and it
was first built for that very purpose.

The entire package system is vectorized using NumPy. This allows both the environments and the agents to run very fast,
especially on large scale experiments.

**Note:** this is not meant to be an entry level framework. Users are expected to be at least somewhat experienced in both
python and reinforcement learning.

**Features**

*Included in package:*

- Environment abstract class to wrap any environment with gym-like methods
- Interface system to allow communication between agents and environment (for example to make a certain environment not fully observable)
- Customizable experiments, structured in training/validation and test volleys, plotting the following metrics:
    - Average total reward (reward in one episode) over training, validation and test volleys and episodes (per each volley)
    - Average scaled reward (reward per step) over training, validation and test volleys and episodes (per each volley)
    - Standard deviation of said total and scaled rewards over training, validation and test volleys
- Utility functions to easily run experiments (even multiple iterations of one type of experiment)
- Many state-of-the-art algorithms, including:
    - Tabular Temporal Difference Q-Learning, SARSA, Expected SARSA with Prioritized Experience Replay memory buffer
    - Deep Temporal Difference Q-Learning (DQN), SARSA, Expected SARSA with Prioritized Experience Replay memory buffer
    - Double Deep Temporal Difference Q-Learning (DDQN) with Prioritized Experience Replay memory buffer
    - Dueling Temporal Difference Q-Learning (DDDQN) with Prioritized Experience Replay memory buffer
    - Vanilla Policy Gradient (VPG) with General Advantage Estimate (GAE) buffer using rewards-to-go
    - Proximal Policy Optimization (PPO) with General Advantage Estimate (GAE) buffer using rewards-to-go and early stopping
    - Deep Deterministic Policy Gradient (DDPG) with simple FIFO buffer
- Default agents for all the included algorithms, including:
    - Q-Learning both tabular and approximated by DNNs with Epsilon Greedy, Boltzmann and Dirichlet exploration policies
    - SARSA both tabular and approximated by DNNs with Epsilon Greedy, Boltzmann and Dirichlet exploration policies
    - Expected SARSA both tabular and approximated by DNNs with Epsilon Greedy, Boltzmann and Dirichlet exploration policies
    - Vanilla Policy Gradient and Proximal Policy Optimization
    - Deep Deterministic Policy Gradient with Gaussian Noise
- Customizable config class to easily define the layers of the networks (when applicable)

*Not included in package:*

- Extensive set of benchmarks for each default agent using the OpenAI gym environment as a reference
- OpenAI gym environment wrappers and benchmark experiment as implementation samples

For additional example of usage of this framework, take a look a these GitHub pages (using old versions of the framework):

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

- pip install usienarl[tensorflow-gpu] to install tensorflow-gpu version
- pip install usienarl[tensorflow] to install tensorflow cpu version

**Note:** failure in specifying the extra require will cause tensorflow to not be installed, and as such the library won't
be usable at all. For instance, this is not allowed, *unless you already have tensorflow installed*:

- pip install usienarl

Tensorflow support ranges from 1.10 to 1.15. Please report any kind of incompatibilities.
Some future warnings could be issued by tensorflow if they are not removed (look at the benchmarks to see how to remove them).

If you want to improve/modify/extends the framework, or even just try my own benchmarks at home, download or clone
the `git repository <https://github.com/InsaneMonster/USienaRL>`_.
You are welcome to open issues or participate in the project. Note that the benchmarks are usually run using tensorflow-gpu.

**Requirements**

Besides Tensorflow, with this package also the following packages will be installed in your environment:

- NumPy
- SciPy
- Pandas
- Matplotlib

**How to use**

For a simple use case, refer to benchmark provided in the `repository <https://github.com/InsaneMonster/USienaRL>`_.
For advanced use, refer to the built-in documentation and to the provided source code in the `repository <https://github.com/InsaneMonster/USienaRL>`_.

**Current issues**

DDPG is experimental. It should work but its performance is not consistent.
Experiment iterations are not implemented as well as they could be. I'm still thinking about a better implementation.

**Changelog**

**v0.7.0**:

- Vectorized the entire package.
- Added DDPG algorithm and agent.
- Improved all algorithms. They are now clearer, optimized and more intuitive in their implementation.
- Improved all default agents. They are now clearer, optimized and more intuitive in their implementation.
- Improved the experiment. Now volleys are classes and all collected data and metrics are easily accessible through attributes.
- Plots are now saved both per volley and per episode inside each volley
- Refactored utility functions
- A huge list of minor fixes and improvements
- Largely improved summaries and built-in documentation

**v0.7.1**:

- Fixed a return value of the tabular SARSA algorithm
- Fixed interface now able to translate environment actions to agent actions on discrete spaces when possible actions list have different sizes
- Added default routine to translate environment possible actions to agent possible actions (not fully optimized yet)
- Largely improved and optimized pass-through interface routine to translate between environment possible actions to agent possible actions
- Possible actions are now list instead of arrays. Built-in documentation is updated accordingly.

**v0.7.2**:
- Fixed some missing logs at the end of test in the experiment class
- Fixed plots x-axis using sometimes float-values instead of int-values

**CREDITS**

Luca Pasqualini at `SAILab <http://sailab.diism.unisi.it/people/luca-pasqualini/>`_ - University of Siena.