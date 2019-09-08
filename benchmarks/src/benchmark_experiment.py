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

# Import usienarl

from usienarl import Experiment, Environment, Agent, Interface


class BenchmarkExperiment(Experiment):
    """
    Benchmark experiment for OpenAI gym environments.

    It only uses a validation threshold to both validate and test. If validation is passed, the experiment is considered
    automatically successful.
    """

    def __init__(self,
                 name: str,
                 validation_threshold: float,
                 environment: Environment,
                 agent: Agent,
                 interface: Interface = None):
        # Define benchmark experiment attributes
        self._validation_threshold: float = validation_threshold
        # Generate the base experiment
        super(BenchmarkExperiment, self).__init__(name, environment, agent, interface)

    def _is_validated(self,
                      logger: logging.Logger,
                      last_average_validation_total_reward: float, last_average_validation_scaled_reward: float,
                      last_average_training_total_reward: float, last_average_training_scaled_reward: float,
                      last_validation_volley_rewards: [], last_training_volley_rewards: []) -> bool:
        # Check if average validation reward (score) is over validation threshold
        if last_average_validation_total_reward >= self._validation_threshold:
            return True
        return False

    def _display_test_cycle_metrics(self,
                                    last_test_cycle_average_total_reward: float,
                                    last_test_cycle_average_scaled_reward: float,
                                    last_test_cycle_rewards: []):
        pass

    def _is_successful(self,
                       logger: logging.Logger,
                       average_test_total_reward: float, average_test_scaled_reward: float,
                       max_test_total_reward: float, max_test_scaled_reward: float,
                       last_average_validation_total_reward: float, last_average_validation_scaled_reward: float,
                       last_average_training_total_reward: float, last_average_training_scaled_reward: float,
                       test_cycles_rewards: [],
                       last_validation_volley_rewards: [], last_training_volley_rewards: []) -> bool:
        # Check if last validation reward (score) was above threshold
        if last_average_validation_total_reward >= self._validation_threshold:
            return True
        return False
