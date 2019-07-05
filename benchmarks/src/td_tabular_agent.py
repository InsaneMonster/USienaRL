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
from usienarl.models.policy_optimization import VanillaPolicyGradient


class TDTabularAgent(Agent):

    def __init__(self,
                 model: VanillaPolicyGradient,
                 name: str):
        # Define tensorflow model
        self.model: VanillaPolicyGradient = model
        # Define internal agent attributes
        self._current_value_estimate = None
        # Generate base agent
        super(TDTabularAgent, self).__init__(name)

    def initialize(self,
                   logger: logging.Logger,
                   session):
        """
        Overridden method of Agent class: check its docstring for further information.
        """
        # Initialize the model
        self.model.initialize(logger, session)

    def update(self,
               logger: logging.Logger,
               session,
               current_episode: int, total_episodes: int, current_step: int):
        """
        Overridden method of Agent class: check its docstring for further information.
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _generate(self,
                  logger: logging.Logger,
                  experiment_scope: str) -> bool:
        """
        Overridden method of Agent class: check its docstring for further information.
        """
        # Reset internal agent attributes
        self._current_value_estimate = None
        # Generate the model and return a flag stating if generation was successful
        return self.model.generate(logger, experiment_scope + "/" + self.name,
                                   self.interface.observation_space_type, self.interface.observation_space_shape,
                                   self.interface.agent_action_space_type, self.interface.agent_action_space_shape)

    def _decide_pre_train(self,
                          logger: logging.Logger,
                          session,
                          agent_observation_current: numpy.ndarray):
        """
        Overridden method of Agent class: check its docstring for further information.
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _decide_train(self,
                      logger: logging.Logger,
                      session,
                      agent_observation_current: numpy.ndarray):
        """
        Overridden method of Agent class: check its docstring for further information.
        """
        # Just predict the action since the model is inherently exploring
        action, self._current_value_estimate = self.model.predict(session, agent_observation_current)

    def _decide_inference(self,
                          logger: logging.Logger,
                          session,
                          agent_observation_current: numpy.ndarray):
        """
        Overridden method of Agent class: check its docstring for further information.
        """
        # Predict the action with the model
        action, _ = self.model.predict(session, agent_observation_current)

    def _save_step_pre_train(self,
                             logger: logging.Logger,
                             session,
                             agent_observation_current: numpy.ndarray,
                             agent_action,
                             reward: float,
                             agent_observation_next: numpy.ndarray,
                             episode_done: bool):
        """
        Overridden method of Agent class: check its docstring for further information.
        """
        # Empty method, it should be implemented on a child class basis
        pass

    def _save_step_train(self,
                         logger: logging.Logger,
                         session,
                         agent_observation_current: numpy.ndarray,
                         agent_action,
                         reward: float,
                         agent_observation_next: numpy.ndarray,
                         episode_done: bool):
        """
        Overridden method of Agent class: check its docstring for further information.
        """
        #
        self.model.buffer.store_train(agent_observation_current, agent_action, reward, self._current_value_estimate)
        #
        if episode_done:
            self.model.buffer.finish_path(reward)

    def get_trainable_variables(self,
                                experiment_scope: str):
        """
        Overridden method of Agent class: check its docstring for further information.
        """
        # Return the trainable variables of the agent model in the given experiment scope
        return self.model.get_trainable_variables(experiment_scope + "/" + self.name)

    @property
    def pre_train_episodes(self) -> int:
        """
        Overridden property of Agent class: check its docstring for further information.
        """
        # Empty property with default return value, it should be implemented on a child class basis if required
        return 0
