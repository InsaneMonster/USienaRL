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
        # Generate the base experiment
        super(BenchmarkExperiment, self).__init__(name, environment, agent, interface)
        # Define internal attributes
        self._validation_threshold: float = validation_threshold

    def _is_validated(self,
                      logger: logging.Logger) -> bool:
        # Check if average validation reward (score) is over validation threshold
        if self.validation_volley.avg_total_reward >= self._validation_threshold:
            return True
        return False

    def _is_successful(self,
                       logger: logging.Logger) -> bool:
        # Check if validated
        return self.validated
