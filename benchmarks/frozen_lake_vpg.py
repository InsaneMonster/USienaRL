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
from usienarl.agents.vpg_agent import VPGAgent

# Import required src
# Require error handling to support both deployment and pycharm versions

try:
    from src.openai_gym_environment import OpenAIGymEnvironment
    from src.benchmark_experiment import BenchmarkExperiment
except ImportError:
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
    # Return the model
    return VanillaPolicyGradient("model", discount_factor,
                                 learning_rate_policy, learning_rate_advantage,
                                 value_steps_per_update, config, lambda_parameter)


def _define_agent(model: VanillaPolicyGradient, explore: bool = True) -> VPGAgent:
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
    return VPGAgent("vpg_agent", model, updates_per_training_volley,
                    alpha, dirichlet_trade_off_min, dirichlet_trade_off_max, dirichlet_trade_off_update)


def run(workspace: str,
        experiment_iterations: int,
        render_training: bool, render_validation: bool, render_test: bool):
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Frozen Lake environment:
    #       - general success threshold to consider the training and the experiment successful is 0.78 over 100 episodes according to OpenAI guidelines
    environment_name: str = 'FrozenLake-v0'
    success_threshold: float = 0.78
    # Generate the OpenAI environment
    environment: OpenAIGymEnvironment = OpenAIGymEnvironment(environment_name)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    # Define model
    inner_model: VanillaPolicyGradient = _define_vpg_model(nn_config)
    # Define agents
    ppo_agent_default: VPGAgent = _define_agent(inner_model, False)
    ppo_agent_explore: VPGAgent = _define_agent(inner_model, True)
    # Define experiments
    experiment_default: BenchmarkExperiment = BenchmarkExperiment("experiment_default", success_threshold, environment,
                                                                  ppo_agent_default)
    experiment_explore: BenchmarkExperiment = BenchmarkExperiment("experiment_dirichlet", success_threshold, environment,
                                                                  ppo_agent_explore)
    # Define experiments data
    testing_episodes: int = 100
    test_cycles: int = 10
    training_episodes: int = 5000
    validation_episodes: int = 100
    max_training_episodes: int = 1000000
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



