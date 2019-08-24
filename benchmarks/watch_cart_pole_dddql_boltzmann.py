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

from usienarl import Config, LayerType, watch_experiment, command_line_parse
from usienarl.td_models import DuelingDeepQLearning
from usienarl.exploration_policies import BoltzmannExplorationPolicy

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


def _define_dddqn_model(config: Config) -> DuelingDeepQLearning:
    # Define attributes
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    buffer_capacity: int = 1000
    minimum_sample_probability: float = 0.01
    random_sample_trade_off: float = 0.6
    importance_sampling_value_increment: float = 0.4
    importance_sampling_value: float = 0.001
    error_clip: bool = False
    # Return the _model
    return DuelingDeepQLearning("model",
                                learning_rate, discount_factor,
                                buffer_capacity,
                                minimum_sample_probability, random_sample_trade_off,
                                importance_sampling_value, importance_sampling_value_increment,
                                config, error_clip)


def _define_boltzmann_exploration_policy() -> BoltzmannExplorationPolicy:
    # Define attributes
    temperature_max: float = 1.0
    temperature_min: float = 0.001
    temperature_decay: float = 0.001
    # Return the explorer
    return BoltzmannExplorationPolicy(temperature_max, temperature_min, temperature_decay)


def _define_boltzmann_agent(model: DuelingDeepQLearning, exploration_policy: BoltzmannExplorationPolicy) -> DuelingDeepQLearningAgent:
    # Define attributes
    weight_copy_step_interval: int = 25
    batch_size: int = 100
    # Return the agent
    return DuelingDeepQLearningAgent("dddqn_boltzmann_agent", model, exploration_policy, weight_copy_step_interval, batch_size)


if __name__ == "__main__":
    # Parse the command line arguments
    checkpoint_path, iteration_number, cuda_devices, render = command_line_parse(True)
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
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu])
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu])
    # Define model
    inner_model: DuelingDeepQLearning = _define_dddqn_model(nn_config)
    # Define exploration policy
    boltzmann_exploration_policy: BoltzmannExplorationPolicy = _define_boltzmann_exploration_policy()
    # Define agent
    dddqn_boltzmann_agent: DuelingDeepQLearningAgent = _define_boltzmann_agent(inner_model, boltzmann_exploration_policy)
    # Define experiment
    experiment_boltzmann: BenchmarkExperiment = BenchmarkExperiment("b_experiment_" + str(iteration_number) if iteration_number != -1 else "b_experiment", success_threshold, environment,
                                                                    dddqn_boltzmann_agent)
    # Define experiment data
    testing_episodes: int = 100
    test_cycles: int = 10
    episode_length_max: int = 100000
    # Run boltzmann experiment
    watch_experiment(experiment_boltzmann,
                     testing_episodes, test_cycles,
                     episode_length_max,
                     render,
                     logger, checkpoint_path)


