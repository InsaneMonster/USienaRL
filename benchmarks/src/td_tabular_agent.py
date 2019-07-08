#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
# USienaRL is licensed under a MIT License.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/MIT>.

# Import packages

import numpy
import logging

# Import required src

from usienarl import Agent
from usienarl.models.temporal_difference import Tabular


class TDTabularAgent(Agent):
    """
    TODO: summary

    """

    def __init__(self,
                 model: Tabular,
                 exploration_rate_max: float, exploration_rate_min: float, exploration_rate_decay: float,
                 name: str):
        # Define tensorflow model
        self.model: Tabular = model
        # Define internal agent attributes
        self._exploration_rate: float = None
        self._exploration_rate_max: float = exploration_rate_max
        self._exploration_rate_min: float = exploration_rate_min
        self._exploration_rate_decay: float = exploration_rate_decay
        # Generate base agent
        super(TDTabularAgent, self).__init__(name)

    def initialize(self,
                   logger: logging.Logger,
                   session):
        # Reset internal agent attributes
        self._exploration_rate = self._exploration_rate_max
        # Initialize the model
        self.model.initialize(logger, session)

    def update(self,
               logger: logging.Logger,
               session,
               current_episode: int, total_episodes: int, current_step: int):
        # Empty method, it should be implemented on a child class basis
        pass

    def _generate(self,
                  logger: logging.Logger,
                  scope: str) -> bool:
        # Generate the model and return a flag stating if generation was successful
        return self.model.generate(logger, scope + "/" + self._name,
                                   self.interface.observation_space_type, self.interface.observation_space_shape,
                                   self.interface.agent_action_space_type, self.interface.agent_action_space_shape)

    def _decide_train(self,
                      logger: logging.Logger,
                      session,
                      agent_observation_current: int):
        # Predict an action using the model or a random action with epsilon greedy exploration
        if self._exploration_rate > 0.0 and numpy.random.rand(1) < self._exploration_rate:
            action: int = self.environment.get_random_action(logger, session)
        else:
            action: int = self.model.predict(session, agent_observation_current)
        # Return the predicted action
        return action

    def _decide_pre_train(self,
                          logger: logging.Logger,
                          session,
                          agent_observation_current: numpy.ndarray):
        # Not needed by this agent
        pass

    def _decide_inference(self,
                          logger: logging.Logger,
                          session,
                          agent_observation_current: int):
        # Predict the action with the model
        action: int = self.model.predict(session, agent_observation_current)
        # Return the predicted action
        return action

    def _save_step_pre_train(self,
                             logger: logging.Logger,
                             session,
                             agent_observation_current: numpy.ndarray,
                             agent_action,
                             reward: float,
                             agent_observation_next: numpy.ndarray,
                             episode_done: bool):
        # Not needed by this agent
        pass

    def _save_step_train(self,
                         logger: logging.Logger,
                         session,
                         agent_observation_current: numpy.ndarray,
                         agent_action,
                         reward: float,
                         agent_observation_next: numpy.ndarray,
                         episode_done: bool):
        # TODO: add to model buffer with n-step
        self.model.buffer.store_train(agent_observation_current, agent_action, reward, self._current_value_estimate)
        # Decrease exploration rate value
        if episode_done:
            self._exploration_rate = max(self._exploration_rate - self._exploration_rate_decay, self._exploration_rate_min)

    def get_trainable_variables(self,
                                scope: str):
        # Return the trainable variables of the agent model in the given experiment scope
        return self.model.get_trainable_variables(scope + "/" + self._name)
