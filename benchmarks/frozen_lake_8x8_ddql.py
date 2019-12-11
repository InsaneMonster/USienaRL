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
from usienarl.td_models import DoubleDeepQLearning
from usienarl.agents import DoubleDeepQLearningAgentEpsilonGreedy, DoubleDeepQLearningAgentBoltzmann, DoubleDeepQLearningAgentDirichlet

# Import required src
# Require error handling to support both deployment and pycharm versions

try:
    from src.openai_gym_environment import OpenAIGymEnvironment
    from src.benchmark_experiment import BenchmarkExperiment
except ImportError:
    from benchmarks.src.openai_gym_environment import OpenAIGymEnvironment
    from benchmarks.src.benchmark_experiment import BenchmarkExperiment

# Define utility functions to run the experiment


def _define_ddqn_model(config: Config, error_clip: bool = True) -> DoubleDeepQLearning:
    # Define attributes
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    buffer_capacity: int = 1000
    minimum_sample_probability: float = 0.01
    random_sample_trade_off: float = 0.6
    importance_sampling_value_increment: float = 0.4
    importance_sampling_value: float = 0.001
    # Return the _model
    return DoubleDeepQLearning("model_mse" if not error_clip else "model_huber",
                               learning_rate, discount_factor,
                               buffer_capacity,
                               minimum_sample_probability, random_sample_trade_off,
                               importance_sampling_value, importance_sampling_value_increment,
                               config, error_clip)


def _define_epsilon_greedy_agent(model: DoubleDeepQLearning) -> DoubleDeepQLearningAgentEpsilonGreedy:
    # Define attributes
    weight_copy_step_interval: int = 25
    batch_size: int = 100
    exploration_rate_max: float = 1.0
    exploration_rate_min: float = 0.001
    exploration_rate_decay: float = 0.001
    # Return the agent
    return DoubleDeepQLearningAgentEpsilonGreedy("ddqn_agent", model, weight_copy_step_interval, batch_size,
                                                 exploration_rate_max, exploration_rate_min, exploration_rate_decay)


def _define_boltzmann_agent(model: DoubleDeepQLearning) -> DoubleDeepQLearningAgentBoltzmann:
    # Define attributes
    weight_copy_step_interval: int = 25
    batch_size: int = 100
    temperature_max: float = 1.0
    temperature_min: float = 0.001
    temperature_decay: float = 0.001
    # Return the agent
    return DoubleDeepQLearningAgentBoltzmann("ddqn_agent", model, weight_copy_step_interval, batch_size,
                                             temperature_max, temperature_min, temperature_decay)


def _define_dirichlet_agent(model: DoubleDeepQLearning) -> DoubleDeepQLearningAgentDirichlet:
    # Define attributes
    weight_copy_step_interval: int = 25
    batch_size: int = 100
    alpha: float = 1.0
    dirichlet_trade_off_min: float = 0.5
    dirichlet_trade_off_max: float = 1.0
    dirichlet_trade_off_update: float = 0.001
    # Return the agent
    return DoubleDeepQLearningAgentDirichlet("ddqn_agent", model, weight_copy_step_interval, batch_size,
                                             alpha, dirichlet_trade_off_min, dirichlet_trade_off_max, dirichlet_trade_off_update)


def run(workspace: str,
        experiment_iterations: int,
        render_training: bool, render_validation: bool, render_test: bool):
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Frozen Lake environment:
    #       - general success threshold to consider the training and the experiment successful is 0.78 over 100 episodes according to OpenAI guidelines
    environment_name: str = 'FrozenLake8x8-v0'
    success_threshold: float = 0.78
    # Generate the OpenAI environment
    environment: OpenAIGymEnvironment = OpenAIGymEnvironment(environment_name)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    # Define model
    inner_model: DoubleDeepQLearning = _define_ddqn_model(nn_config)
    # Define agents
    ddqn_agent_epsilon_greedy: DoubleDeepQLearningAgentEpsilonGreedy = _define_epsilon_greedy_agent(inner_model)
    ddqn_agent_boltzmann: DoubleDeepQLearningAgentBoltzmann = _define_boltzmann_agent(inner_model)
    ddqn_agent_dirichlet: DoubleDeepQLearningAgentDirichlet = _define_dirichlet_agent(inner_model)
    # Define experiments
    experiment_epsilon_greedy: BenchmarkExperiment = BenchmarkExperiment("experiment_epsilon_greedy", success_threshold, environment,
                                                                         ddqn_agent_epsilon_greedy)
    experiment_boltzmann: BenchmarkExperiment = BenchmarkExperiment("experiment_boltzmann", success_threshold, environment,
                                                                    ddqn_agent_boltzmann)
    experiment_dirichlet: BenchmarkExperiment = BenchmarkExperiment("experiment_dirichlet", success_threshold, environment,
                                                                    ddqn_agent_dirichlet)
    # Define experiments data
    testing_episodes: int = 100
    test_cycles: int = 10
    training_episodes: int = 100
    validation_episodes: int = 100
    max_training_episodes: int = 10000
    episode_length_max: int = 100
    plot_sample_density_training_episodes: int = 10
    plot_sample_density_validation_episodes: int = 10
    # Run experiments
    run_experiment(experiment_epsilon_greedy,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_training, render_validation, render_test,
                   workspace, __file__,
                   logger, None, experiment_iterations,
                   None,
                   plot_sample_density_training_episodes, plot_sample_density_validation_episodes)
    run_experiment(experiment_boltzmann,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_training, render_validation, render_test,
                   workspace, __file__,
                   logger, None, experiment_iterations,
                   None,
                   plot_sample_density_training_episodes, plot_sample_density_validation_episodes)
    run_experiment(experiment_dirichlet,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_training, render_validation, render_test,
                   workspace, __file__,
                   logger, None, experiment_iterations,
                   None,
                   plot_sample_density_training_episodes, plot_sample_density_validation_episodes)


if __name__ == "__main__":
    # Remove tensorflow deprecation warnings
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Run this experiment
    run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
