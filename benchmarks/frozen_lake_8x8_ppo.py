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
from usienarl.po_models import ProximalPolicyOptimization
from usienarl.agents.ppo_agent import PPOAgent

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


def _define_ppo_model(config: Config) -> ProximalPolicyOptimization:
    # Define attributes
    learning_rate_policy: float = 0.0003
    learning_rate_advantage: float = 0.001
    discount_factor: float = 0.99
    value_steps_per_update: int = 80
    policy_steps_per_update: int = 80
    lambda_parameter: float = 0.97
    clip_ratio: float = 0.2
    target_kl_divergence: float = 0.01
    # Return the model
    return ProximalPolicyOptimization("model", discount_factor,
                                      learning_rate_policy, learning_rate_advantage,
                                      value_steps_per_update, policy_steps_per_update,
                                      config,
                                      lambda_parameter,
                                      clip_ratio,
                                      target_kl_divergence)


def _define_agent(model: ProximalPolicyOptimization, explore: bool = True) -> PPOAgent:
    # Define attributes
    updates_per_training_volley: int = 10
    alpha: float = 1.0
    dirichlet_trade_off_max: float = 1.0
    dirichlet_trade_off_update: float = 0.001
    # Remove exploration by setting min dirichlet trade-off to its max
    if explore:
        dirichlet_trade_off_min: float = 0.5
    else:
        dirichlet_trade_off_min: float = dirichlet_trade_off_max
    # Return the agent
    return PPOAgent("ppo_agent", model, updates_per_training_volley,
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
    environment_name: str = 'FrozenLake8x8-v0'
    success_threshold: float = 0.78
    success_threshold_refactored: float = -16
    # Generate the OpenAI environment
    environment: OpenAIGymEnvironment = OpenAIGymEnvironment(environment_name)
    # Generate the refactored environment
    environment_refactored: FrozenLakeRefactoredEnvironment = FrozenLakeRefactoredEnvironment(environment_name)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [64, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    # Define model
    inner_model: ProximalPolicyOptimization = _define_ppo_model(nn_config)
    # Define agents
    ppo_agent_default: PPOAgent = _define_agent(inner_model, False)
    ppo_agent_explore: PPOAgent = _define_agent(inner_model, True)
    # Define experiments
    experiment_default: BenchmarkExperiment = BenchmarkExperiment("experiment_default", success_threshold, environment,
                                                                  ppo_agent_default)
    experiment_explore: BenchmarkExperiment = BenchmarkExperiment("experiment_dirichlet", success_threshold, environment,
                                                                  ppo_agent_explore)
    # Define refactored experiments
    experiment_refactored_default: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored_default", success_threshold_refactored,
                                                                             environment_refactored,
                                                                             ppo_agent_default)
    experiment_refactored_explore: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored_dirichlet", success_threshold_refactored,
                                                                             environment_refactored,
                                                                             ppo_agent_explore)
    # Define experiments data
    testing_episodes: int = 100
    test_cycles: int = 10
    training_episodes: int = 5000
    validation_episodes: int = 100
    max_training_episodes: int = 100000
    episode_length_max: int = 100
    plot_sample_density_training_episodes: int = 500
    plot_sample_density_validation_episodes: int = 10
    # Run experiments
    run_experiment(experiment_default,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_training, render_validation, render_test,
                   workspace, __file__,
                   logger, None, experiment_iterations,
                   None,
                   plot_sample_density_training_episodes, plot_sample_density_validation_episodes)
    run_experiment(experiment_explore,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_training, render_validation, render_test,
                   workspace, __file__,
                   logger, None, experiment_iterations,
                   None,
                   plot_sample_density_training_episodes, plot_sample_density_validation_episodes)
    # Run refactored experiments
    run_experiment(experiment_refactored_default,
                   training_episodes,
                   max_training_episodes, episode_length_max,
                   validation_episodes,
                   testing_episodes, test_cycles,
                   render_training, render_validation, render_test,
                   workspace, __file__,
                   logger, None, experiment_iterations,
                   None,
                   plot_sample_density_training_episodes, plot_sample_density_validation_episodes)
    run_experiment(experiment_refactored_explore,
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
