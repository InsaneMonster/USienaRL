USienaRL
*********

This is a *python 3.6* and above library for Reinforcement Learning (RL) experiments.

The idea behind this library is to generate an intuitive yet versatile system to generate RL agents, experiments, models, etc.
The library is modular, and allow for easy experiment iterations and validation. It is currently used in my research and it
was built first for that purpose.

**Note:** this is not meant to be an entry level framework. Users are expected to be at least somewhat experienced in both
python and reinforcement learning. The framework is easy to extend but almost all the agent-related and environment-related
work should be done by yourself assembling pre-made or custom parts.

**Features**

*Included in package:*

- Model, Agent, Environment, Experiment abstract classes with their interrelationships hidden inside the implementation
- Customizable experiments in which, besides the default metrics, additional metrics can be used to validate or pass the experiment:
    - Metrics of experiments are always related to rewards obtained (per-step or per-episode) and training episodes (usually the minimum the better)
- Utility functions to run the same experiment in multiple equal iterations with automated folder setup and organization, registering the following metrics:
    - Average total reward (reward in one episode) over training, validation and test
    - Average scaled reward (reward per step) over training, validation and test
    - Standard deviation of said total and scaled rewards over training, validation and test
    - Mean and standard deviation of average total and scaled reward over test in all experiment iterations (if more than one)
    - Mean and standard deviation of maximum total and scaled reward over test in all experiment iterations (if more than one)
    - Mean and standard deviation of minimum training episodes over test in all experiment iterations (if more than one)
    - Experiment iteration achieving best results in each one of the metrics described above
    - Easy to use .csv file with all the results for each experiment iteration
    - Plots of total and scaled rewards over both all training and validation episodes, as well as std of both and average episode length, all saved as .png files
- Many state-of-the-art algorithms already implemented in pre-defined models, including:
    - Tabular Temporal Difference Q-Learning, SARSA, Expected SARSA with Prioritized Experience Replay memory buffer
    - Deep Temporal Difference Q-Learning (DQN), SARSA, Expected SARSA with Prioritized Experience Replay memory buffer
    - Double Deep Temporal Difference Q-Learning (DDQN) with Prioritized Experience Replay memory buffer
    - Dueling Temporal Difference Q-Learning (DDDQN) with Prioritized Experience Replay memory buffer
    - Vanilla Policy Gradient (VPG) with General Advantage Estimate (GAE) buffer using rewards-to-go
    - Proximal Policy Optimization (PPO) with General Advantage Estimate (GAE) buffer using rewards-to-go and early stopping
- Many state-of-the-art exploration policies embedded into the default agents, including:
    - Epsilon Greedy with tunable decay rate, start value and end value for temporal difference agents
    - Boltzmann sampling with tunable temperature decay rate, start value and end value for temporal difference agents
    - Dirichlet distribution with tunable alpha and x parameters for policy optimization agents acting on discrete states
- Config class to define the hidden layers of all Tensorflow graphs (including the CNN):
    - Customization of layer types also possible through extension of the class
- Default Pass-Through interface class to allow communications between agents and environments
- Additive action mask for all the algorithms supporting it (only discrete action sets):
    - The mask supports two values: -infinity (mask) and 0.0 (pass-through)
    - If not supplied, the mask is by default full pass-through
- Default agents for all the included algorithms, including:
    - Q-Learning both tabular and approximated by DNNs with Epsilon Greedy and Boltzmann exploration policies
    - SARSA both tabular and approximated by DNNs with Epsilon Greedy and Boltzmann exploration policies
    - Expected SARSA both tabular and approximated by DNNs with Epsilon Greedy and Boltzmann exploration policies
    - Vanilla Policy Gradient and Proximal Policy Optimization with optional Dirichlet exploration policy for discrete action spaces

*Not included in package:*

- Extensive set of benchmarks for each default agent using the OpenAI gym environment as a reference
- OpenAI gym environment wrapper and benchmark experiment classes as good sample on how to implement environments or experiment classes

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

Some summaries and docs could not be fully clear, homogeneous or up-to-date. This will be fixed in upcoming releases.

From the save-restore standpoint it could be useful to implement an easy way to pass a metagraph without the need to redefine the entire agent, for example serializing all the agent-related data.
The same can be said for the experiment as a whole, with all needed data serialized and ready to be "watched" with just one simple script.
Ideally serialize data could be loaded from folders referring to agents, environments and the experiments as a whole.

Beside that, an experiment can right now work only in a specific environment.
It could be interesting to test multiple environments both from a curriculum learning perspective (it can still be done using multiple subsequent experiments, however) and from a generalization perspective (train one one, validate on another, etc).
A way to check if environments are compatible one another would be required too to have that implemented.

**Changelog**

- Version 0.6.2: Hot-fixed Vanilla Policy Gradient agent dirichlet exploration policy not working correctly
- Version 0.6.1: Hot-fixed deep SARSA agent and tabular SARSA agent dirichlet exploration policy not working correctly

- Added default agents to the package: Tabular QL, Tabular SARSA, Tabular ExpectedSARSA, DQN, DDQN, DDDQN, DSARSA, DExpectedSARSA, VPG, PPO
- Removed exploration policies, now embedded into agents. This allows for further customization when making your own agents
- Added Dirichlet exploration policy to overall all agents supporting it
- Added average episode length (measured in steps) as plot, improved plots differentiations and coloring
- Improved Tensorboard summaries for policy optimization algorithms
- Added progress reports over training and validation volleys with additional information
- Default agents now come with additional information regarding what are they doing when training/updating
- Some minor fix and improvements

**CREDITS**

Luca Pasqualini at `SAILab <http://sailab.diism.unisi.it/people/luca-pasqualini/>`_ - University of Siena.