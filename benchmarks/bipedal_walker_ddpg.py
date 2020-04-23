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

from usienarl import Config, LayerType
from usienarl.utils import run_experiment, command_line_parse
from usienarl.models import DeepDeterministicPolicyGradient
from usienarl.agents import DDPGAgent

# Import required src
# Require error handling to support both deployment and pycharm versions

try:
    from src.openai_gym_environment import OpenAIGymEnvironment
    from src.benchmark_experiment import BenchmarkExperiment
except ImportError:
    from benchmarks.src.openai_gym_environment import OpenAIGymEnvironment
    from benchmarks.src.benchmark_experiment import BenchmarkExperiment

# Define utility functions to run the experiment


def _define_ddpg_model(actor_config: Config, critic_config: Config) -> DeepDeterministicPolicyGradient:
    # Define attributes
    learning_rate_policy: float = 1e-9
    learning_rate_q_values: float = 1e-9
    discount_factor: float = 0.99
    polyak_value: float = 0.999
    buffer_capacity: int = 1000000
    # Return the model
    return DeepDeterministicPolicyGradient("model", actor_config, critic_config,
                                           learning_rate_policy, learning_rate_q_values,
                                           discount_factor, polyak_value, buffer_capacity)


def _define_agent(model: DeepDeterministicPolicyGradient) -> DDPGAgent:
    # Define attributes
    update_every_steps: int = 50
    summary_save_every_steps: int = 500
    batch_size: int = 100
    noise_scale_max: float = 0.3
    noise_scale_min: float = 0.0
    noise_decay: float = 1e-5
    # Return the agent
    return DDPGAgent("ddpg_agent", model, update_every_steps, summary_save_every_steps, batch_size,
                     noise_scale_max, noise_scale_min, noise_decay)


def run(workspace: str,
        experiment_iterations: int,
        render_training: bool, render_validation: bool, render_test: bool):
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Mountain Car environment:
    #       - general success threshold to consider the training and the experiment successful is -110.0 over 100 episodes according to OpenAI guidelines
    environment_name: str = 'BipedalWalker-v2'
    success_threshold: float = 300.0
    # Generate the OpenAI environment
    environment: OpenAIGymEnvironment = OpenAIGymEnvironment(environment_name)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [300, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    # nn_config.add_hidden_layer(LayerType.batch_normalization, [])
    nn_config.add_hidden_layer(LayerType.dense, [600, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    # nn_config.add_hidden_layer(LayerType.batch_normalization, [])
    # Define model
    inner_model: DeepDeterministicPolicyGradient = _define_ddpg_model(actor_config=nn_config, critic_config=nn_config)
    # Define agent
    ddpg_agent: DDPGAgent = _define_agent(inner_model)
    # Define experiment
    experiment_default: BenchmarkExperiment = BenchmarkExperiment("experiment_default", success_threshold, environment,
                                                                  ddpg_agent)
    # Define experiment data
    parallel: int = 1
    training_episodes: int = 50
    validation_episodes: int = 100
    training_validation_volleys: int = 100
    test_episodes: int = 100
    test_volleys: int = 10
    episode_length_max: int = 1000
    # Run experiment
    run_experiment(logger=logger, experiment=experiment_default,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=True,
                   render_during_test=render_test,
                   iterations=1, saves_to_keep=3, plots_dpi=150)


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
