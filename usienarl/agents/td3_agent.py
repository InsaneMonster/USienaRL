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
from usienarl.models import TwinDelayedDDPG


class TD3Agent(Agent):
    """
    Twin Delayed DDPG agent with simple Gaussian noise exploration policy.
    It is supplied with a TD3 model.

    The update every steps value define after how many steps an update should be performed. When updating, a number
    of updated is executed which is proportional to the number of elapsed steps. Policy will only be updated once every
    policy delay times for each update of q-stream.
    The noise is a gaussian white noise and it is necessary to perform exploration during training.
    """

    def __init__(self,
                 name: str,
                 model: TwinDelayedDDPG,
                 update_every_steps: int = 50,
                 policy_delay: int = 2,
                 summary_save_every_steps: int = 500,
                 batch_size: int = 100,
                 noise_scale_max: float = 0.1, noise_scale_min: float = 0.0,
                 noise_scale_decay: float = 1e-6):
        # Define internal attributes
        self._model: TwinDelayedDDPG = model
        self._update_every_steps: int = update_every_steps
        self._policy_delay: int = policy_delay
        self._noise_scale_max: float = noise_scale_max
        self._noise_scale_min: float = noise_scale_min
        self._noise_scale_decay: float = noise_scale_decay
        self._summary_save_every_steps: int = summary_save_every_steps
        self._batch_size: int = batch_size
        # Define empty attributes
        self._last_update_step: int = 0
        self._last_summary_save_step: int = 0
        self._noise_scale: float or None = None
        # Generate base agent
        super(TD3Agent, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape: (),
                  agent_action_space_type: SpaceType, agent_action_space_shape: ()) -> bool:
        # Generate the model and check if it's successful, stop if not successful
        return self._model.generate(logger, self._scope + "/" + self._name,
                                    self._parallel,
                                    observation_space_type, observation_space_shape,
                                    agent_action_space_type, agent_action_space_shape)

    def initialize(self,
                   logger: logging.Logger,
                   session):
        # Reset attributes
        self._last_update_step = 0
        self._last_summary_save_step = 0
        # Initialize the model
        self._model.initialize(logger, session)
        # Initialize the noise scale value for exploration
        self._noise_scale = self._noise_scale_max
        # Run the weight copy operation to uniform main and target networks
        self._model.weights_copy(session)

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   interface: Interface,
                   agent_observation_current: numpy.ndarray,
                   warmup_step: int, warmup_episode: int):
        # Act randomly
        return interface.sample_agent_action(logger, session)

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  interface: Interface,
                  agent_observation_current: numpy.ndarray,
                  train_step: int, train_episode: int):
        # Get the possible actions (action boundaries)
        possible_actions: numpy.ndarray = interface.possible_agent_actions(logger, session)
        lower_bound: numpy.ndarray = possible_actions[:, 0]
        upper_bound: numpy.ndarray = possible_actions[:, 1]
        # Get the best action predicted by the model
        action = self._model.get_best_action(session, agent_observation_current, possible_actions)
        # Add a certain Gaussian noise according to its current scale
        noise: numpy.ndarray = numpy.zeros((self._parallel, *self._agent_action_space_shape), dtype=float)
        for i in range(self._parallel):
            noise[i] = numpy.random.randn(*self._agent_action_space_shape)
        action += self._noise_scale * noise
        return numpy.clip(action, lower_bound, upper_bound)

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      interface: Interface,
                      agent_observation_current: numpy.ndarray,
                      inference_step: int, inference_episode: int):
        # Return the best action predicted by the model
        return self._model.get_best_action(session, agent_observation_current, interface.possible_agent_actions(logger, session))

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
        # Save the current step in the buffer
        self._model.buffer.store(agent_observation_current.copy(), agent_action.copy(), reward.copy(), agent_observation_next.copy(), episode_done.copy())

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
        # Save the current step in the buffer
        self._model.buffer.store(agent_observation_current.copy(), agent_action.copy(), reward.copy(), agent_observation_next.copy(), episode_done.copy())
        # Execute update only after a certain number of steps each time
        if train_step - self._last_update_step >= self._update_every_steps:
            q_stream_loss_total: float = 0.0
            policy_stream_loss_total: float = 0.0
            policy_updates_number: int = 0
            # Update for as many step as you have been waiting
            for step in range(train_step - self._last_update_step):
                # Execute the update and store q-stream loss
                batch: [] = self._model.buffer.get(self._batch_size)
                q_stream_loss = self._model.update_q_values(session, batch)
                # Update total q-stream loss metric
                q_stream_loss_total += q_stream_loss
                # Execute the policy update every policy delay steps (update target network together with it)
                if step % self._policy_delay == 0:
                    policy_stream_loss = self._model.update_policy(session, batch)
                    # Update total policy loss metric
                    policy_stream_loss_total += policy_stream_loss
                    policy_updates_number += 1
                    # Update target network
                    self._model.weights_update(session)
            # Generate the tensorflow summary on the average metrics
            if train_step - self._last_summary_save_step >= self._summary_save_every_steps:
                # Generate the tensorflow summary on the loss and add it to the writer
                summary = tensorflow.Summary()
                policy_stream_loss_average: float = policy_stream_loss_total / policy_updates_number
                q_stream_loss_average: float = q_stream_loss_total / (train_step - self._last_update_step)
                summary.value.add(tag="policy_stream_loss", simple_value=policy_stream_loss_average)
                summary.value.add(tag="q_stream_loss", simple_value=q_stream_loss_average)
                # Update the summary at the absolute current step
                self._summary_writer.add_summary(summary, train_step)
                # Update last summary save step
                self._last_summary_save_step = train_step
            # Update last update step
            self._last_update_step = train_step

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
        # Update the buffer at the end of trajectory
        self._model.buffer.finish_trajectory()

    def complete_episode_train(self,
                               logger: logging.Logger,
                               session,
                               interface: Interface,
                               last_step_reward: numpy.ndarray,
                               episode_total_reward: numpy.ndarray,
                               train_step: int, train_episode: int):
        # Update the buffer at the end of trajectory
        self._model.buffer.finish_trajectory()
        # Decrease the noise scale by its decay value
        self._noise_scale = max(self._noise_scale_min, self._noise_scale - self._noise_scale_decay)

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
