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
import numpy
import tensorflow

# Import usienarl

from usienarl import Agent, Interface, SpaceType
from usienarl.models import VanillaPolicyGradient


class VPGAgent(Agent):
    """
    Vanilla Policy Gradient agent without specific exploration policies (beside sampling which is automatically
    done by the model). It is supplied with a VPG model.

    The number of episodes to play before updating is a crucial parameter to stabilize learning.
    Increase it if the environment has high variance. The default value of 100 is tailored to training volleys of length
    around 1000.
    Note that the number of episodes actually executed between two updates of the model may actually differ a bit from
    that value with respect to the parallelization used.
    """

    def __init__(self,
                 name: str,
                 model: VanillaPolicyGradient,
                 update_every_episodes: int = 100):
        # Define internal attributes
        self._model: VanillaPolicyGradient = model
        self._update_every_episodes: int = update_every_episodes
        # Define empty attributes
        self._last_update_episode: int = 0
        self._current_value_estimate = None
        self._dirichlet_trade_off: float or None = None
        # Generate base agent
        super(VPGAgent, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape,
                  agent_action_space_type: SpaceType, agent_action_space_shape) -> bool:
        # Generate the model and check if it's successful, stop if not successful
        return self._model.generate(logger, self._scope + "/" + self._name,
                                    self._parallel,
                                    observation_space_type, observation_space_shape,
                                    agent_action_space_type, agent_action_space_shape)

    def initialize(self,
                   logger: logging.Logger,
                   session):
        # Reset attributes
        self._last_update_episode = 0
        self._current_value_estimate = None
        # Initialize the model
        self._model.initialize(logger, session)

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   interface: Interface,
                   agent_observation_current,
                   warmup_step: int, warmup_episode: int):
        pass

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  interface: Interface,
                  agent_observation_current: numpy.ndarray,
                  train_step: int, train_episode: int):
        # Act according to the model best prediction (a sample from the probability distribution)
        # Note: it is inherently exploring
        action, self._current_value_estimate, _ = self._model.sample_action(session, agent_observation_current, interface.possible_agent_actions(logger, session))
        # Return the exploration action
        return action

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      interface: Interface,
                      agent_observation_current: numpy.ndarray,
                      inference_step: int, inference_episode: int):
        # Predict the action with the model by sampling from its probability distribution
        action, _, _ = self._model.sample_action(session, agent_observation_current, interface.possible_agent_actions(logger, session))
        # Return the predicted action
        return action

    def complete_step_warmup(self,
                             logger: logging.Logger,
                             session,
                             interface: Interface,
                             agent_observation_current: numpy.ndarray,
                             agent_action: numpy.ndarray,
                             reward: numpy.ndarray,
                             episode_done: numpy.ndarray,
                             agent_observation_next: numpy.ndarray,
                             warmup_step: int, warmup_episode: int):
        pass

    def complete_step_train(self,
                            logger: logging.Logger,
                            session,
                            interface: Interface,
                            agent_observation_current: numpy.ndarray,
                            agent_action: numpy.ndarray,
                            reward: numpy.ndarray,
                            episode_done: numpy.ndarray,
                            agent_observation_next: numpy.ndarray,
                            train_step: int, train_episode: int):
        # Save the current step in the buffer together with the current value estimate
        self._model.buffer.store(agent_observation_current.copy(), agent_action.copy(), reward.copy(), episode_done.copy(), self._current_value_estimate.copy())

    def complete_step_inference(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
                                agent_observation_current: numpy.ndarray,
                                agent_action: numpy.ndarray,
                                reward: numpy.ndarray,
                                episode_done: numpy.ndarray,
                                agent_observation_next: numpy.ndarray,
                                inference_step: int, inference_episode: int):
        pass

    def complete_episode_warmup(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
                                last_step_reward: numpy.ndarray,
                                episode_total_reward: numpy.ndarray,
                                warmup_step: int, warmup_episode: int):
        pass

    def complete_episode_train(self,
                               logger: logging.Logger,
                               session,
                               interface: Interface,
                               last_step_reward: numpy.ndarray,
                               episode_total_reward: numpy.ndarray,
                               train_step: int, train_episode: int):
        # Update the buffer at the end of trajectory
        self._model.buffer.finish_trajectory(last_step_reward)
        # Execute update only after a certain number of trajectories each time
        if train_episode - self._last_update_episode >= self._update_every_episodes:
            # Execute the update and store policy and value loss
            buffer: [] = self._model.buffer.get()
            logger.info("Updating the model with a buffer of " + str(len(buffer[0])) + " samples")
            policy_stream_loss, value_stream_loss, approximated_entropy = self._model.update(session, buffer)
            # Generate the tensorflow summary
            summary = tensorflow.Summary()
            summary.value.add(tag="policy_stream_loss", simple_value=policy_stream_loss)
            summary.value.add(tag="value_stream_loss", simple_value=value_stream_loss)
            summary.value.add(tag="approximated_entropy", simple_value=approximated_entropy)
            # Update the summary at the absolute current step if required
            if self._summary_writer is not None:
                self._summary_writer.add_summary(summary, train_step)
            # Update last update episode
            self._last_update_episode = train_episode

    def complete_episode_inference(self,
                                   logger: logging.Logger,
                                   session,
                                   interface: Interface,
                                   last_step_reward: numpy.ndarray,
                                   episode_total_reward: numpy.ndarray,
                                   inference_step: int, inference_episode: int):
        pass

    @property
    def saved_variables(self):
        return self._model.trainable_variables

    @property
    def warmup_steps(self) -> int:
        return self._model.warmup_steps

    @property
    def update_every_episodes(self) -> int:
        """
        The number of trajectories to play before updating.
        """
        return self._update_every_episodes
