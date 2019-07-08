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

from usienarl import Agent, SpaceType
from usienarl.models.policy_optimization import VanillaPolicyGradient


class VPGAgent(Agent):
    """
    TODO: summary

    """

    def __init__(self,
                 model: VanillaPolicyGradient,
                 updates_per_training_volley: int,
                 name: str):
        # Define VPG agent attributes
        self.updates_per_training_volley: int = updates_per_training_volley
        # Define tensorflow _model
        self._model: VanillaPolicyGradient = model
        # Define internal agent attributes
        self._current_value_estimate = None
        self._current_policy_loss = None
        self._current_value_loss = None
        # Generate base agent
        super(VPGAgent, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape,
                  action_space_type: SpaceType, action_space_shape) -> bool:
        # Generate the _model and return a flag stating if generation was successful
        return self._model.generate(logger, self._scope + "/" + self._name,
                                    observation_space_type, observation_space_shape,
                                    action_space_type, action_space_shape)

    def initialize(self,
                   logger: logging.Logger,
                   session):
        # Reset internal agent attributes
        self._current_value_estimate = None
        self._current_policy_loss = None
        self._current_value_loss = None
        # Initialize the _model
        self._model.initialize(logger, session)

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   agent_observation_current):
        pass

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  agent_observation_current):
        # Predict the action since the _model is inherently exploring
        # Also store the current value estimate to feed the buffer later-on
        action, self._current_value_estimate = self._model.get_best_action(session, agent_observation_current)
        # Return the predicted action
        return action

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      agent_observation_current):
        # Predict the action with the _model
        action, _ = self._model.get_best_action(session, agent_observation_current)
        # Return the predicted action
        return action

    def complete_step_warmup(self,
                             logger: logging.Logger,
                             session,
                             agent_observation_current,
                             agent_action,
                             reward: float,
                             agent_observation_next,
                             warmup_step_current: int,
                             warmup_episode_current: int,
                             warmup_episode_volley: int):
        pass

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
        # Save the current step in the buffer together with the current value estimate
        self._model.buffer.store(agent_observation_current, agent_action, reward, self._current_value_estimate)

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
        # Update the buffer at the end of trajectory
        self._model.buffer.finish_path(last_step_reward)
        # Execute update only after a certain number of episodes
        if train_episode_current % (train_episode_volley / self.updates_per_training_volley) == 0 and train_episode_current > 0:
            # Execute the update and store policy and value loss
            summary, self._current_policy_loss, self._current_value_loss = self._model.update(session, self._model.buffer.get())
            # Update the _summary at the absolute current step
            self._summary_writer.add_summary(summary, train_step_absolute)

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
        return self._model.warmup_episodes



