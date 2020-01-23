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

import logging
import os

# Import usienarl

from usienarl import run_experiment, command_line_parse
from usienarl.td_models import TabularExpectedSARSA
from usienarl.agents import TabularExpectedSARSAAgentEpsilonGreedy, TabularExpectedSARSAAgentBoltzmann, TabularExpectedSARSAAgentDirichlet

# Import required src
# Require error handling to support both deployment and pycharm versions

try:
    from src.openai_gym_environment import OpenAIGymEnvironment
    from src.frozen_lake_refactored_environment import FrozenLakeRefactoredEnvironment
    from src.benchmark_experiment import BenchmarkExperiment
except ImportError:
    from benchmarks.src.openai_gym_environment import OpenAIGymEnvironment
    from benchmarks.src.frozen_lake_refactored_environment import FrozenLakeRefactoredEnvironment
    from benchmarks.src.benchmark_experiment import BenchmarkExperiment

# Define utility functions to run the experiment


def _define_tesarsa_model() -> TabularExpectedSARSA:
    # Define attributes
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    buffer_capacity: int = 1000
    minimum_sample_probability: float = 0.01
    random_sample_trade_off: float = 0.6
    importance_sampling_value_increment: float = 0.4
    importance_sampling_value: float = 0.001
    # Return the _model
    return TabularExpectedSARSA("model",
                                learning_rate, discount_factor,
                                buffer_capacity,
                                minimum_sample_probability, random_sample_trade_off,
                                importance_sampling_value, importance_sampling_value_increment)


def _define_epsilon_greedy_agent(model: TabularExpectedSARSA) -> TabularExpectedSARSAAgentEpsilonGreedy:
    # Define attributes
    batch_size: int = 100
    exploration_rate_max: float = 1.0
    exploration_rate_min: float = 0.001
    exploration_rate_decay: float = 0.001
    # Return the agent
    return TabularExpectedSARSAAgentEpsilonGreedy("tesarsa_agent", model, batch_size,
                                                  exploration_rate_max, exploration_rate_min, exploration_rate_decay)


def _define_boltzmann_agent(model: TabularExpectedSARSA) -> TabularExpectedSARSAAgentBoltzmann:
    # Define attributes
    batch_size: int = 100
    temperature_max: float = 1.0
    temperature_min: float = 0.001
    temperature_decay: float = 0.001
    # Return the agent
    return TabularExpectedSARSAAgentBoltzmann("tesarsa_agent", model, batch_size,
                                              temperature_max, temperature_min, temperature_decay)


def _define_dirichlet_agent(model: TabularExpectedSARSA) -> TabularExpectedSARSAAgentDirichlet:
    # Define attributes
    batch_size: int = 100
    alpha: float = 1.0
    dirichlet_trade_off_min: float = 0.5
    dirichlet_trade_off_max: float = 1.0
    dirichlet_trade_off_update: float = 0.001
    # Return the agent
    return TabularExpectedSARSAAgentDirichlet("tesarsa_agent", model,  batch_size,
                                              alpha, dirichlet_trade_off_min, dirichlet_trade_off_max, dirichlet_trade_off_update)


def run(workspace: str,
        experiment_iterations: int,
        render_training: bool, render_validation: bool, render_test: bool):
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Frozen Lake environment:
    #       - general success threshold to consider the training and the experiment successful is 0.78 over 100 episodes according to OpenAI guidelines
    #       - general success threshold for refactored environment is little above (slippery) the minimum number of steps required to reach the goal
    environment_name: str = 'FrozenLake-v0'
    success_threshold: float = 0.78
    success_threshold_refactored: float = -8
    # Generate the OpenAI environment
    environment: OpenAIGymEnvironment = OpenAIGymEnvironment(environment_name)
    # Generate the refactored environment
    environment_refactored: FrozenLakeRefactoredEnvironment = FrozenLakeRefactoredEnvironment(environment_name)
    # Define model
    inner_model: TabularExpectedSARSA = _define_tesarsa_model()
    # Define agents
    tesarsa_agent_epsilon_greedy: TabularExpectedSARSAAgentEpsilonGreedy = _define_epsilon_greedy_agent(inner_model)
    tesarsa_agent_boltzmann: TabularExpectedSARSAAgentBoltzmann = _define_boltzmann_agent(inner_model)
    tesarsa_agent_dirichlet: TabularExpectedSARSAAgentDirichlet = _define_dirichlet_agent(inner_model)
    # Define experiments
    experiment_epsilon_greedy: BenchmarkExperiment = BenchmarkExperiment("experiment_epsilon_greedy", success_threshold, environment,
                                                                         tesarsa_agent_epsilon_greedy)
    experiment_boltzmann: BenchmarkExperiment = BenchmarkExperiment("experiment_boltzmann", success_threshold, environment,
                                                                    tesarsa_agent_boltzmann)
    experiment_dirichlet: BenchmarkExperiment = BenchmarkExperiment("experiment_dirichlet", success_threshold, environment,
                                                                    tesarsa_agent_dirichlet)
    # Define refactored experiments
    experiment_epsilon_greedy_refactored: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored_epsilon_greedy", success_threshold_refactored,
                                                                                    environment_refactored,
                                                                                    tesarsa_agent_epsilon_greedy)
    experiment_boltzmann_refactored: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored_boltzmann", success_threshold_refactored,
                                                                               environment_refactored,
                                                                               tesarsa_agent_boltzmann)
    experiment_dirichlet_refactored: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored_dirichlet", success_threshold_refactored,
                                                                               environment_refactored,
                                                                               tesarsa_agent_dirichlet)
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
                   logger, None, None, experiment_iterations,
                   None,
                   plot_sample_density_training_episodes, plot_sample_density_validation_episodes)
    run_experiment(experiment_boltzmann,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_training, render_validation, render_test,
                   workspace, __file__,
                   logger, None, None, experiment_iterations,
                   None,
                   plot_sample_density_training_episodes, plot_sample_density_validation_episodes)
    run_experiment(experiment_dirichlet,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_training, render_validation, render_test,
                   workspace, __file__,
                   logger, None, None, experiment_iterations,
                   None,
                   plot_sample_density_training_episodes, plot_sample_density_validation_episodes)
    # Run refactored experiments
    run_experiment(experiment_epsilon_greedy_refactored,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_training, render_validation, render_test,
                   workspace, __file__,
                   logger, None, None, experiment_iterations,
                   None,
                   plot_sample_density_training_episodes, plot_sample_density_validation_episodes)
    run_experiment(experiment_boltzmann_refactored,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_training, render_validation, render_test,
                   workspace, __file__,
                   logger, None, None, experiment_iterations,
                   None,
                   plot_sample_density_training_episodes, plot_sample_density_validation_episodes)
    run_experiment(experiment_dirichlet_refactored,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_training, render_validation, render_test,
                   workspace, __file__,
                   logger, None, None, experiment_iterations,
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
