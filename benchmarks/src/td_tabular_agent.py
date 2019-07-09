#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
# USienaRL is licensed under a MIT License.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/MIT>.

# Import packages

import logging

# Import required src

from usienarl import Agent, ExplorationPolicy, SpaceType
from usienarl.models.temporal_difference import Tabular


class TDTabularAgent(Agent):
    """
    TODO: summary

    """

    def __init__(self,
                 name: str,
                 model: Tabular,
                 exploration_policy: ExplorationPolicy):
        # Define tabular agent attributes
        self._model: Tabular = model
        self._exploration_policy: ExplorationPolicy = exploration_policy
        # Define internal agent attributes
        self._current_absolute_errors = None
        self._current_loss = None
        # Generate base agent
        super(TDTabularAgent, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape,
                  action_space_type: SpaceType, action_space_shape) -> bool:
        # Generate the exploration policy and check if it's successful, stop if not successful
        if self._exploration_policy.generate(logger, action_space_type, action_space_shape):
            # Generate the _model and return a flag stating if generation was successful
            return self._model.generate(logger, self._scope + "/" + self._name,
                                        observation_space_type, observation_space_shape,
                                        action_space_type, action_space_shape)
        return False

    def initialize(self,
                   logger: logging.Logger,
                   session):
        # Reset internal agent attributes
        self._current_absolute_errors = None
        self._current_loss = None
        # Initialize the model
        self._model.initialize(logger, session)
        # Initialize the exploration policy
        self._exploration_policy.initialize(logger, session)

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   agent_observation_current):
        # Act randomly
        action = self.env

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  agent_observation_current):
        # Get all actions and the best action predicted by the model
        best_action, all_actions = self._model.get_best_action_and_all_actions(session, agent_observation_current)
        # Act according to the exploration policy
        action = self._exploration_policy.act(session, all_actions, best_action)
        # Return the exploration action
        return action

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      agent_observation_current):
        # Act with the best policy according to the model
        action = self._model.get_best_action(session, agent_observation_current)
        # Return the predicted action
        return action

    def complete_step_warmup(self,
                             logger: logging.Logger,
                             session,
                             agent_observation_current,
                             agent_action, reward: float,
                             agent_observation_next,
                             warmup_step_current: int,
                             warmup_episode_current: int,
                             warmup_episode_volley: int):
        # Save the current step in the buffer
        self._model.buffer.store(agent_observation_current, agent_action, reward, agent_observation_next)

    def complete_step_train(self,
                            logger: logging.Logger,
                            session,
                            agent_observation_current,
                            agent_action,
                            reward: float,
                            agent_observation_next,
                            train_step_current: int, train_step_absolute: int,
                            train_episode_current: int, train_episode_absolute: int,
                            train_episode_volley: int, train_episode_total: int):
        # Save the current step in the buffer
        self._model.buffer.store(agent_observation_current, agent_action, reward, agent_observation_next)

    def complete_step_inference(self,
                                logger: logging.Logger,
                                session,
                                agent_observation_current,
                                agent_action,
                                reward: float,
                                agent_observation_next,
                                inference_step_current: int,
                                inference_episode_current: int,
                                inference_episode_volley: int):
        pass

    def complete_episode_warmup(self,
                                logger: logging.Logger,
                                session,
                                last_step_reward: float,
                                episode_total_reward: float,
                                warmup_episode_current: int,
                                warmup_episode_volley: int):
        pass

    def complete_episode_train(self,
                               logger: logging.Logger,
                               session,
                               last_step_reward: float,
                               episode_total_reward: float,
                               train_step_absolute: int,
                               train_episode_current: int, train_episode_absolute: int,
                               train_episode_volley: int, train_episode_total: int):
        pass

    def complete_episode_inference(self,
                                   logger: logging.Logger,
                                   session,
                                   last_step_reward: float,
                                   episode_total_reward: float,
                                   inference_episode_current: int,
                                   inference_episode_volley: int):
        pass

    @property
    def trainable_variables(self):
        # Return the trainable variables of the agent model in experiment/agent _scope
        return self._model.trainable_variables

    @property
    def warmup_episodes(self) -> int:
        # Return the amount of warmup episodes required by the model
        return self._model.warmup_episodes
