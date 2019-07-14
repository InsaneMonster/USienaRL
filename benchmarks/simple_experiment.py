# Import packages

import sys
import os
import logging
import tensorflow

# Import usienarl package

from usienarl import Config, LayerType, Experiment
from po_models import VanillaPolicyGradient

# Import required src

from benchmarks.src.vpg_agent import VPGAgent
from benchmarks.src.openai_gym_environment import OpenAIGymEnvironment
from benchmarks.src.benchmark_experiment import BenchmarkExperiment


# Define functions

def _define_vpg_model(config: Config) -> VanillaPolicyGradient:
    # Define attributes
    learning_rate_policy: float = 0.0003
    learning_rate_advantage: float = 0.001
    discount_factor: float = 0.99
    value_steps_per_update: int = 80
    lambda_parameter: float = 0.95
    # Return the _model
    return VanillaPolicyGradient("vpg", discount_factor,
                                 learning_rate_policy, learning_rate_advantage,
                                 value_steps_per_update, config, lambda_parameter)


def _define_vpg_agent(model: VanillaPolicyGradient) -> VPGAgent:
    # Define attributes
    updates_per_training_volley: int = 10
    # Return the agent
    return VPGAgent("vpg_agent", model, updates_per_training_volley)


if __name__ == "__main__":
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Cart Pole environment:
    #       - validation success threshold to consider the training completed is 195.0 according to OpenAI guidelines
    #       - test success threshold not required and set to 0 since the environment is considered solved on validation according to OpenAI guidelines
    environment_name: str = 'CartPole-v0'
    validation_success_threshold: float = 195.0
    # Generate the OpenAI environment
    environment: OpenAIGymEnvironment = OpenAIGymEnvironment(environment_name)
    # Define NN for Policy Optimization
    policy_optimization_nn_config: Config = Config()
    policy_optimization_nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu])
    policy_optimization_nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu])
    # Define policy optimization model
    model_vpg: VanillaPolicyGradient = _define_vpg_model(policy_optimization_nn_config)
    # Define policy optimization agent
    agent_vpg: VPGAgent = _define_vpg_agent(model_vpg)
    # Define policy optimization experiment
    experiment: Experiment = BenchmarkExperiment("simple_experiment", validation_success_threshold, environment, agent_vpg)
    # Define common experiment data
    testing_episodes: int = 100
    test_cycles: int = 10
    # Run policy optimization experiment
    training_episodes: int = 1000
    validation_episodes: int = 100
    max_training_episodes: int = 50000
    max_episode_length: int = 1000000
    if experiment.setup("/workspace/summary", "/workspace/metagraph", logger):
        experiment.conduct(training_episodes, validation_episodes,
                           max_training_episodes, max_episode_length,
                           testing_episodes, test_cycles,
                           logger, False, True, True)
