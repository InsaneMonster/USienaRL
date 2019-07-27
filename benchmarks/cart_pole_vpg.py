#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# USienaRL is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import tensorflow
import logging
import os

# Import usienarl

from usienarl import Config, LayerType, run_experiment, command_line_parse
from usienarl.po_models import VanillaPolicyGradient
from usienarl.exploration_policies import EpsilonGreedyExplorationPolicy

# Import required src
# Require error handling to support both deployment and pycharm versions

try:
    from src.vpg_agent import VPGAgent
    from src.openai_gym_environment import OpenAIGymEnvironment
    from src.benchmark_experiment import BenchmarkExperiment
except ImportError:
    from benchmarks.src.vpg_agent import VPGAgent
    from benchmarks.src.openai_gym_environment import OpenAIGymEnvironment
    from benchmarks.src.benchmark_experiment import BenchmarkExperiment

# Define utility functions to run the experiment


def _define_vpg_model(config: Config) -> VanillaPolicyGradient:
    # Define attributes
    learning_rate_policy: float = 0.0003
    learning_rate_advantage: float = 0.001
    discount_factor: float = 0.99
    value_steps_per_update: int = 80
    lambda_parameter: float = 0.95
    # Return the _model
    return VanillaPolicyGradient("model", discount_factor,
                                 learning_rate_policy, learning_rate_advantage,
                                 value_steps_per_update, config, lambda_parameter)


def _define_epsilon_greedy_exploration_policy() -> EpsilonGreedyExplorationPolicy:
    # Define attributes
    exploration_rate_max: float = 1.0
    exploration_rate_min: float = 0.001
    exploration_rate_decay: float = 0.0001
    # Return the explorer
    return EpsilonGreedyExplorationPolicy(exploration_rate_max, exploration_rate_min, exploration_rate_decay)


def _define_epsilon_greedy_agent(model: VanillaPolicyGradient, exploration_policy: EpsilonGreedyExplorationPolicy = None) -> VPGAgent:
    # Define attributes
    updates_per_training_volley: int = 10
    # Return the agent
    return VPGAgent("vpg_egreedy_agent" if exploration_policy is not None else "vpg_agent", model, updates_per_training_volley, exploration_policy)


if __name__ == "__main__":
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Cart Pole environment:
    #       - general success threshold to consider the training and the experiment successful is 195.0 over 100 episodes according to OpenAI guidelines
    environment_name: str = 'CartPole-v0'
    success_threshold: float = 195.0
    # Generate the OpenAI environment
    environment: OpenAIGymEnvironment = OpenAIGymEnvironment(environment_name)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.tanh])
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.tanh])
    # Define model
    inner_model: VanillaPolicyGradient = _define_vpg_model(nn_config)
    # Define exploration_policies
    epsilon_greedy_exploration_policy: EpsilonGreedyExplorationPolicy = _define_epsilon_greedy_exploration_policy()
    # Define agents
    vpg_agent: VPGAgent = _define_epsilon_greedy_agent(inner_model)
    vpg_epsilon_greedy_agent: VPGAgent = _define_epsilon_greedy_agent(inner_model, epsilon_greedy_exploration_policy)
    # Define experiments
    experiment: BenchmarkExperiment = BenchmarkExperiment("experiment", success_threshold, environment,
                                                          vpg_agent)
    experiment_egreedy: BenchmarkExperiment = BenchmarkExperiment("eg_experiment", success_threshold, environment,
                                                                  vpg_epsilon_greedy_agent)
    # Define experiments data
    testing_episodes: int = 100
    test_cycles: int = 10
    training_episodes: int = 1000
    validation_episodes: int = 100
    max_training_episodes: int = 100000
    episode_length_max: int = 100000
    # Run experiment without exploration policy
    run_experiment(experiment,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger, None, experiment_iterations_number)
    # Run experiment with exploration policy
    run_experiment(experiment_egreedy,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger, None, experiment_iterations_number)


