# Import packages

import tensorflow
import logging
import os

# Import usienarl basics

from usienarl import Config, LayerType, run_experiment, command_line_parse

# Import usienarl experiments

from usienarl.experiments import PolicyOptimizationExperiment, QLearningExperiment

# Import usienarl q-learning models

from td_models import DeepQLearning, DoubleDQN, DuelingDeepQLearning

# Import usienarl memories

from usienarl.memories import ExperienceReplay, PrioritizedExperienceReplay
# Import usienarl exploration_policies

from usienarl.exploration_policies import EpsilonGreedyExplorationPolicy, BoltzmannExplorationPolicy

# Import usienarl policy optimization models

from po_models import VanillaPolicyGradient

# Import OpenAI environment

from benchmarks.src.openai_gym_environment import OpenAIGymEnvironment

# Define utility functions to run the experiment


def _define_dqn_model(config: Config) -> DeepQLearning:
    # Define attributes
    weight_copy_step_interval: int = 25
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    # Return the _model
    return DeepQLearning("dqn", learning_rate, discount_factor, config, weight_copy_step_interval)


def _define_ddqn_model(config: Config) -> DoubleDQN:
    # Define attributes
    weight_copy_step_interval: int = 25
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    # Return the _model
    return DoubleDQN("ddqn", learning_rate, discount_factor, config, weight_copy_step_interval)


def _define_dddqn_model(config: Config) -> DuelingDeepQLearning:
    # Define attributes
    weight_copy_step_interval: int = 25
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    # Return the _model
    return DuelingDeepQLearning("dddqn", learning_rate, discount_factor, config, weight_copy_step_interval)


def _define_epsilon_greedy_explorer() -> EpsilonGreedyExplorationPolicy:
    # Define attributes
    exploration_rate_start_value: float = 1.0
    exploration_rate_end_value: float = 0.1
    exploration_rate_value_decay: float = 0.001
    # Return the explorer
    return EpsilonGreedyExplorationPolicy(exploration_rate_start_value, exploration_rate_end_value, exploration_rate_value_decay)


def _define_boltzmann_explorer() -> BoltzmannExplorationPolicy:
    # Define attributes
    exploration_rate_start_value: float = 1.0
    exploration_rate_end_value: float = 0.1
    exploration_rate_value_decay: float = 0.001
    # Return the explorer
    return BoltzmannExplorationPolicy(exploration_rate_start_value, exploration_rate_end_value, exploration_rate_value_decay)


def _define_experience_replay_memory() -> ExperienceReplay:
    # Define attributes
    memory_capacity: int = 500
    pre_train: bool = False
    # Return the memory
    return ExperienceReplay(memory_capacity, pre_train)


def _define_prioritized_experience_replay_memory() -> PrioritizedExperienceReplay:
    # Define attributes
    memory_capacity: int = 10000
    pre_train: bool = True
    minimum_sample_probability: float = 0.01
    random_sample_trade_off: float = 0.6
    importance_sampling_value_increment: float = 0.4
    importance_sampling_starting_value: float = 0.001
    # Return the memory
    return PrioritizedExperienceReplay(memory_capacity, pre_train, minimum_sample_probability, random_sample_trade_off, importance_sampling_value_increment, importance_sampling_starting_value)


def _define_vpg_model(config: Config) -> VanillaPolicyGradient:
    # Define attributes
    learning_rate_policy: float = 0.0003
    learning_rate_advantage: float = 0.001
    discount_factor: float = 0.99
    value_steps_per_update: int = 80
    lambda_parameter: float = 0.95
    # Return the _model
    return VanillaPolicyGradient("vpg", learning_rate_policy, learning_rate_advantage,
                                                             discount_factor, value_steps_per_update, config, lambda_parameter)


def _define_q_learning_experiment_batch(experiments_data: []) -> []:
    # Define the batch of experiments
    experiments_batch: [] = []
    # Unpack data
    for model, suffix, memory, batch_size, explorer in experiments_data:
        experiments_batch.append(QLearningExperiment(environment.name + "_" + model.name + suffix,
                                                     validation_success_threshold, test_success_threshold,
                                                     environment,
                                                     model,
                                                     memory, batch_size,
                                                     explorer))
    # Return the batch of experiments
    return experiments_batch


def _define_policy_optimization_experiment_batch(experiments_data: []) -> []:
    # Define the batch of experiments
    experiments_batch: [] = []
    # Unpack data
    for model, suffix, updates_per_training_interval in experiments_data:
        experiments_batch.append(PolicyOptimizationExperiment(environment.name + "_" + model.name + suffix,
                                                              validation_success_threshold, test_success_threshold,
                                                              environment,
                                                              model, updates_per_training_interval))
    # Return the batch of experiments
    return experiments_batch


if __name__ == "__main__":
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Mountain Car environment:
    #       - validation success threshold to consider the training completed is -110.0 according to OpenAI guidelines
    #       - test success threshold not required and set to 0 since the environment is considered solved on validation according to OpenAI guidelines
    environment_name: str = 'MountainCar-v0'
    validation_success_threshold: float = -110.0
    test_success_threshold: float = 0.0
    # Generate the OpenAI environment
    environment: OpenAIGymEnvironment = OpenAIGymEnvironment(environment_name)
    # Define NN for Q-Learning
    q_learning_nn_config: Config = Config()
    q_learning_nn_config.add_hidden_layer(LayerType.dense, [50, tensorflow.nn.relu])
    q_learning_nn_config.add_hidden_layer(LayerType.dense, [50, tensorflow.nn.relu])
    # Define NN for Policy Optimization
    policy_optimization_nn_config: Config = Config()
    policy_optimization_nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.tanh])
    policy_optimization_nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.tanh])
    # Define Q-Learning models
    model_dqn: DeepQLearning = _define_dqn_model(q_learning_nn_config)
    model_ddqn: DoubleDQN = _define_ddqn_model(q_learning_nn_config)
    model_dddqn: DuelingDeepQLearning = _define_dddqn_model(q_learning_nn_config)
    # Define exploration_policies
    epsilon_greedy_explorer: EpsilonGreedyExplorationPolicy = _define_epsilon_greedy_explorer()
    boltzmann_explorer: BoltzmannExplorationPolicy = _define_boltzmann_explorer()
    # Define memories
    experience_replay: ExperienceReplay = _define_experience_replay_memory()
    prioritized_experience_replay: PrioritizedExperienceReplay = _define_prioritized_experience_replay_memory()
    # Define Q-Learning experiments
    # Format: _model, suffix, memory, batch_size and explorer
    q_learning_experiments_data: [] = [(model_dddqn, "_per_b", prioritized_experience_replay, 100, boltzmann_explorer)]
    q_learning_experiments: [] = _define_q_learning_experiment_batch(q_learning_experiments_data)
    # Define policy optimization models
    model_vpg: VanillaPolicyGradient = _define_vpg_model(policy_optimization_nn_config)
    # Define Policy Optimization experiments
    # Format: _model, suffix, updates_per_training_interval
    policy_optimization_experiments_data: [] = [(model_vpg, "", 100)]
    policy_optimization_experiments: [] = _define_policy_optimization_experiment_batch(policy_optimization_experiments_data)
    # Define common experiment data
    testing_episodes: int = 100
    test_cycles: int = 10
    # Run Q-Learning experiments
    training_episodes: int = 250
    validation_episodes: int = 100
    max_training_episodes: int = 15000
    run_experiment(q_learning_experiments, experiment_iterations_number,
                   training_episodes, max_training_episodes, validation_episodes,
                   testing_episodes, test_cycles,
                   False,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger)
    # Run Policy Optimization experiments
    training_episodes: int = 1000
    validation_episodes: int = 100
    max_training_episodes: int = 50000
    run_experiment(policy_optimization_experiments, experiment_iterations_number,
                   training_episodes, max_training_episodes, validation_episodes,
                   testing_episodes, test_cycles,
                   False,
                   render_during_training, render_during_validation, render_during_test,
                   workspace_path, __file__,
                   logger)



