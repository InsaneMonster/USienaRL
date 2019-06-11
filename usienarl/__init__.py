# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# LICENSE NOTICE
#
# USienaRL (University of Siena Reinforcement Learning) (c) by Luca Pasqualini - SAILab - University of Siena
#
# USienaRL is licensed under a
# Creative Commons Attribution-ShareAlike 3.0 Unported License.
#
# You should have received a copy of the license along with this
# work.  If not, see <http://creativecommons.org/licenses/by-sa/3.0/>.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Experiments

from .experiment import Experiment
from .experiments.policy_optimization_experiment import PolicyOptimizationExperiment
from .experiments.q_learning_experiment import QLearningExperiment

# Environment and space types

from .environment import Environment
from .environment import SpaceType

# Generic abstract models and model configurations

from .model import Model
from .config import Config

# Policy optimization models

from .models.policy_optimization_model import PolicyOptimizationModel
from .models.policy_optimization.vanilla_policy_gradient import VanillaPolicyGradient

# Q-Learning models

from .models.q_learning_model import QLearningModel
from .models.q_learning.q_table import QTable
from .models.q_learning.dqn_naive import DQNNaive
from .models.q_learning.dqn import DQN
from .models.q_learning.double_dqn import DoubleDQN
from .models.q_learning.dueling_double_dqn import DuelingDoubleDQN

# Explorer modules

from .explorer import Explorer
from .explorers.epsilon_greedy import EpsilonGreedyExplorer
from .explorers.boltzmann import BoltzmannExplorer

# Memory modules

from .memory import Memory
from .memories.experience_replay import ExperienceReplay
from .memories.prioritized_experience_replay import PrioritizedExperienceReplay

# Visualizer

from .visualizer import Visualizer

# Functions

from .functions import command_line_parse, run_experiments
