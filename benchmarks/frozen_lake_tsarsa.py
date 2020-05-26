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

from usienarl.utils import command_line_parse, run_experiment
from usienarl.models import TabularSARSA
from usienarl.agents import TabularSARSAAgentEpsilonGreedy, TabularSARSAAgentBoltzmann, TabularSARSAAgentDirichlet

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


def _define_tsarsa_model() -> TabularSARSA:
    # Define attributes
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    buffer_capacity: int = 1000
    minimum_sample_probability: float = 1e-2
    random_sample_trade_off: float = 0.6
    importance_sampling_value_increment: float = 0.4
    importance_sampling_value: float = 1e-3
    # Return the _model
    return TabularSARSA("model",
                        buffer_capacity,
                        learning_rate, discount_factor,
                        minimum_sample_probability, random_sample_trade_off,
                        importance_sampling_value, importance_sampling_value_increment)


def _define_epsilon_greedy_agent(model: TabularSARSA) -> TabularSARSAAgentEpsilonGreedy:
    # Define attributes
    summary_save_step_interval: int = 500
    batch_size: int = 100
    exploration_rate_max: float = 1.0
    exploration_rate_min: float = 1e-3
    exploration_rate_decay: float = 1e-3
    # Return the agent
    return TabularSARSAAgentEpsilonGreedy("tsarsa_agent", model, summary_save_step_interval, batch_size,
                                          exploration_rate_max, exploration_rate_min, exploration_rate_decay)


def _define_boltzmann_agent(model: TabularSARSA) -> TabularSARSAAgentBoltzmann:
    # Define attributes
    summary_save_step_interval: int = 500
    batch_size: int = 100
    temperature_max: float = 1.0
    temperature_min: float = 1e-3
    temperature_decay: float = 1e-3
    # Return the agent
    return TabularSARSAAgentBoltzmann("tsarsa_agent", model, summary_save_step_interval, batch_size,
                                      temperature_max, temperature_min, temperature_decay)


def _define_dirichlet_agent(model: TabularSARSA) -> TabularSARSAAgentDirichlet:
    # Define attributes
    summary_save_step_interval: int = 500
    batch_size: int = 100
    alpha: float = 1.0
    dirichlet_trade_off_min: float = 0.5
    dirichlet_trade_off_max: float = 1.0
    dirichlet_trade_off_update: float = 0.001
    # Return the agent
    return TabularSARSAAgentDirichlet("tsarsa_agent", model, summary_save_step_interval, batch_size,
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
    inner_model: TabularSARSA = _define_tsarsa_model()
    # Define agents
    tsarsa_agent_epsilon_greedy: TabularSARSAAgentEpsilonGreedy = _define_epsilon_greedy_agent(inner_model)
    tsarsa_agent_boltzmann: TabularSARSAAgentBoltzmann = _define_boltzmann_agent(inner_model)
    tsarsa_agent_dirichlet: TabularSARSAAgentDirichlet = _define_dirichlet_agent(inner_model)
    # Define experiments
    experiment_epsilon_greedy: BenchmarkExperiment = BenchmarkExperiment("experiment_epsilon_greedy", success_threshold, environment,
                                                                         tsarsa_agent_epsilon_greedy)
    experiment_boltzmann: BenchmarkExperiment = BenchmarkExperiment("experiment_boltzmann", success_threshold, environment,
                                                                    tsarsa_agent_boltzmann)
    experiment_dirichlet: BenchmarkExperiment = BenchmarkExperiment("experiment_dirichlet", success_threshold, environment,
                                                                    tsarsa_agent_dirichlet)
    # Define refactored experiments
    experiment_epsilon_greedy_refactored: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored_epsilon_greedy", success_threshold_refactored,
                                                                                    environment_refactored,
                                                                                    tsarsa_agent_epsilon_greedy)
    experiment_boltzmann_refactored: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored_boltzmann", success_threshold_refactored,
                                                                               environment_refactored,
                                                                               tsarsa_agent_boltzmann)
    experiment_dirichlet_refactored: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored_dirichlet", success_threshold_refactored,
                                                                               environment_refactored,
                                                                               tsarsa_agent_dirichlet)
    # Define experiments data
    saves_to_keep: int = 1
    plots_dpi: int = 150
    parallel: int = 10
    training_episodes: int = 100
    validation_episodes: int = 100
    training_validation_volleys: int = 20
    test_episodes: int = 100
    test_volleys: int = 10
    episode_length_max: int = 100
    # Run experiments
    run_experiment(logger=logger, experiment=experiment_epsilon_greedy,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    run_experiment(logger=logger, experiment=experiment_boltzmann,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    run_experiment(logger=logger, experiment=experiment_dirichlet,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    # Run refactored experiments
    run_experiment(logger=logger, experiment=experiment_epsilon_greedy_refactored,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    run_experiment(logger=logger, experiment=experiment_boltzmann_refactored,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    run_experiment(logger=logger, experiment=experiment_dirichlet_refactored,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)


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

