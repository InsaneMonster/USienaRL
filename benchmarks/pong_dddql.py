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
import math

# Import usienarl

from usienarl import Config, LayerType, run_experiment, command_line_parse
from usienarl.td_models import DuelingDeepQLearning
from usienarl.exploration_policies import EpsilonGreedyExplorationPolicy, BoltzmannExplorationPolicy

# Import required src
# Require error handling to support both deployment and pycharm versions

try:
    from src.dueling_deep_q_learning_agent import DuelingDeepQLearningAgent
    from src.openai_gym_environment import OpenAIGymEnvironment
    from src.benchmark_experiment import BenchmarkExperiment
except ImportError:
    from benchmarks.src.dueling_deep_q_learning_agent import DuelingDeepQLearningAgent
    from benchmarks.src.openai_gym_environment import OpenAIGymEnvironment
    from benchmarks.src.benchmark_experiment import BenchmarkExperiment

# Define utility functions to run the experiment


def _define_dddqn_model(config: Config, error_clip: bool) -> DuelingDeepQLearning:
    # Define attributes
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    buffer_capacity: int = 10000
    minimum_sample_probability: float = 0.01
    random_sample_trade_off: float = 0.6
    importance_sampling_value_increment: float = 0.4
    importance_sampling_value: float = 0.001
    # Return the _model
    return DuelingDeepQLearning("model_mse" if not error_clip else "model_huber",
                                learning_rate, discount_factor,
                                buffer_capacity,
                                minimum_sample_probability, random_sample_trade_off,
                                importance_sampling_value, importance_sampling_value_increment,
                                config, error_clip)


def _define_epsilon_greedy_exploration_policy() -> EpsilonGreedyExplorationPolicy:
    # Define attributes
    exploration_rate_max: float = 1.0
    exploration_rate_min: float = 0.001
    exploration_rate_decay: float = 0.001
    # Return the explorer
    return EpsilonGreedyExplorationPolicy(exploration_rate_max, exploration_rate_min, exploration_rate_decay)


def _define_boltzmann_exploration_policy() -> BoltzmannExplorationPolicy:
    # Define attributes
    temperature_max: float = 1.0
    temperature_min: float = 0.001
    temperature_decay: float = 0.001
    # Return the explorer
    return BoltzmannExplorationPolicy(temperature_max, temperature_min, temperature_decay)


def _define_epsilon_greedy_agent(model: DuelingDeepQLearning, exploration_policy: EpsilonGreedyExplorationPolicy) -> DuelingDeepQLearningAgent:
    # Define attributes
    weight_copy_step_interval: int = 1000
    batch_size: int = 32
    # Return the agent
    return DuelingDeepQLearningAgent("dddqn_egreedy_agent", model, exploration_policy, weight_copy_step_interval, batch_size)


def _define_boltzmann_agent(model: DuelingDeepQLearning, exploration_policy: BoltzmannExplorationPolicy) -> DuelingDeepQLearningAgent:
    # Define attributes
    weight_copy_step_interval: int = 1000
    batch_size: int = 32
    # Return the agent
    return DuelingDeepQLearningAgent("dddqn_boltzmann_agent", model, exploration_policy, weight_copy_step_interval, batch_size)


if __name__ == "__main__":
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Breakout Atari environment: a score of 35.0 (don't know if too high actually...)
    environment_name: str = 'PongDeterministic-v4'
    success_threshold: float = 35.0
    # Generate the OpenAI environment
    environment: OpenAIGymEnvironment = OpenAIGymEnvironment(environment_name)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.convolution_2D, [32, 8, (4, 4), 'valid', 'channels_last', (1, 1), tensorflow.nn.relu])
    nn_config.add_hidden_layer(LayerType.convolution_2D, [64, 4, (2, 2), 'valid', 'channels_last', (1, 1), tensorflow.nn.relu])
    nn_config.add_hidden_layer(LayerType.convolution_2D, [64, 3, (1, 1), 'valid', 'channels_last', (1, 1), tensorflow.nn.relu])
    nn_config.add_hidden_layer(LayerType.flatten, [])
    nn_config.add_hidden_layer(LayerType.dense, [512, tensorflow.nn.relu])
    # Define models
    inner_model_huber: DuelingDeepQLearning = _define_dddqn_model(nn_config, True)
    inner_model_mse: DuelingDeepQLearning = _define_dddqn_model(nn_config, False)
    # Define exploration_policies
    epsilon_greedy_exploration_policy: EpsilonGreedyExplorationPolicy = _define_epsilon_greedy_exploration_policy()
    boltzmann_exploration_policy: BoltzmannExplorationPolicy = _define_boltzmann_exploration_policy()
    # Define agents
    dddqn_epsilon_greedy_agent_huber: DuelingDeepQLearningAgent = _define_epsilon_greedy_agent(inner_model_huber, epsilon_greedy_exploration_policy)
    dddqn_boltzmann_agent_huber: DuelingDeepQLearningAgent = _define_boltzmann_agent(inner_model_huber, boltzmann_exploration_policy)
    dddqn_epsilon_greedy_agent_mse: DuelingDeepQLearningAgent = _define_epsilon_greedy_agent(inner_model_mse, epsilon_greedy_exploration_policy)
    dddqn_boltzmann_agent_mse: DuelingDeepQLearningAgent = _define_boltzmann_agent(inner_model_mse, boltzmann_exploration_policy)
    # Define experiments
    experiment_egreedy_huber: BenchmarkExperiment = BenchmarkExperiment("eg_experiment_huber", success_threshold, environment,
                                                                        dddqn_epsilon_greedy_agent_huber)
    experiment_boltzmann_huber: BenchmarkExperiment = BenchmarkExperiment("b_experiment_huber", success_threshold, environment,
                                                                          dddqn_boltzmann_agent_huber)
    experiment_egreedy_mse: BenchmarkExperiment = BenchmarkExperiment("eg_experiment_mse", success_threshold, environment,
                                                                      dddqn_epsilon_greedy_agent_mse)
    experiment_boltzmann_mse: BenchmarkExperiment = BenchmarkExperiment("b_experiment_mse", success_threshold, environment,
                                                                        dddqn_boltzmann_agent_mse)
    # Define experiments data
    testing_episodes: int = 100
    test_cycles: int = 10
    training_episodes: int = 100
    validation_episodes: int = 100
    max_training_episodes: int = 100000
    episode_length_max: int = 10000000
    # Run epsilon greedy experiments
    run_experiment(experiment_egreedy_huber,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger, None, experiment_iterations_number)
    run_experiment(experiment_egreedy_mse,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger, None, experiment_iterations_number)
    # Run boltzmann experiments
    run_experiment(experiment_boltzmann_huber,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger, None, experiment_iterations_number)
    run_experiment(experiment_boltzmann_mse,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger, None, experiment_iterations_number)
