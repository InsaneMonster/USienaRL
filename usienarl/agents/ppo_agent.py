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

# Import usienarl

from usienarl import Agent, Interface, SpaceType
from usienarl.po_models import ProximalPolicyOptimization


class PPOAgent(Agent):
    """
    Proximal Policy Optimization agent with Dirichlet sampling exploration policies for discrete action spaces.
    It is supplied with a PPO model. By default exploration is disabled. Change the trade-off parameters to enable it.

    The updates per training volley define how many updates should be execute by the model in each train volley. For
    example if the volley is 1000 episodes long and the number of updates per volley is 10, the agent will be updated
    every 100 episodes.

    If exploration policy is enabled, during training, agent samples from the probability distribution of its computed
    values and the dirichlet probability distribution with given alpha, according to a trade-off parameter which is
    updated at the end of each episode.
    """

    def __init__(self,
                 name: str,
                 model: ProximalPolicyOptimization,
                 updates_per_training_volley: int,
                 alpha: float = 1.0,
                 dirichlet_trade_off_min: float = 1.0, dirichlet_trade_off_max: float = 1.0,
                 dirichlet_trade_off_update: float = 0.001):
        # Define agent attributes
        self.updates_per_training_volley: int = updates_per_training_volley
        self._model: ProximalPolicyOptimization = model
        self._alpha: float = alpha
        self._dirichlet_trade_off_min: float = dirichlet_trade_off_min
        self._dirichlet_trade_off_max: float = dirichlet_trade_off_max
        self._dirichlet_trade_off_update: float = dirichlet_trade_off_update
        # Define internal agent attributes
        self._current_value_estimate = None
        self._current_log_likelihood = None
        self._current_policy_loss = None
        self._current_value_loss = None
        self._dirichlet_trade_off: float = None
        # Generate base agent
        super(PPOAgent, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape,
                  agent_action_space_type: SpaceType, agent_action_space_shape) -> bool:
        # Generate the model and check if it's successful, stop if not successful
        return self._model.generate(logger, self._scope + "/" + self._name,
                                    observation_space_type, observation_space_shape,
                                    agent_action_space_type, agent_action_space_shape)

    def initialize(self,
                   logger: logging.Logger,
                   session):
        # Reset internal agent attributes
        self._current_value_estimate = None
        self._current_log_likelihood = None
        self._current_policy_loss = None
        self._current_value_loss = None
        # Initialize the model
        self._model.initialize(logger, session)
        # Reset trade-off to its starting value (the min)
        self._dirichlet_trade_off = self._dirichlet_trade_off_min

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   interface: Interface,
                   agent_observation_current):
        pass

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  interface: Interface,
                  agent_observation_current):
        # Check if trade-off value is equal to 1.0:
        if self._dirichlet_trade_off >= 1.0:
            # Act according to the model best prediction (a sample from the probability distribution)
            # Note: it is still inherently exploring
            action, self._current_value_estimate, self._current_log_likelihood = self._model.sample_action(session, agent_observation_current)
        else:
            # Act according to dirichlet approach: first get probability distribution over all the actions predicted by the model
            prior_probabilities = self._model.get_action_probabilities(session, agent_observation_current)
            # Then generate a dirichlet distribution (d) with parameter alpha
            dirichlet_probabilities = numpy.random.dirichlet(self._alpha * numpy.ones(prior_probabilities.size), 1).flatten()
            # Get a random action value (random output) using x * p + (1 - x) * d as probability distribution where x is the trade-off
            output = self._dirichlet_trade_off * prior_probabilities + (1 - self._dirichlet_trade_off) * dirichlet_probabilities
            # Make sure output sums up to 1.0
            output = output / output.sum()
            action_value = numpy.random.choice(output, p=output)
            # Set the chosen action as the index of such chosen action value
            action = numpy.argmax(output == action_value)
            # Get the value and the log-likelihood of the model at the current observed state with the chosen action
            self._current_value_estimate, self._current_log_likelihood = self._model.get_value_and_log_likelihood(session, action, agent_observation_current)
        # Return the exploration action
        return action

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      interface: Interface,
                      agent_observation_current):
        # Predict the action with the model by sampling from its probability distribution
        action, _, _ = self._model.sample_action(session, agent_observation_current)
        # Return the predicted action
        return action

    def complete_step_warmup(self,
                             logger: logging.Logger,
                             session,
                             interface: Interface,
                             agent_observation_current,
                             agent_action,
                             reward: float,
                             agent_observation_next,
                             warmup_step_current: int,
                             warmup_episode_current: int,
                             warmup_steps_volley: int):
        pass

    def complete_step_train(self,
                            logger: logging.Logger,
                            session,
                            interface: Interface,
                            agent_observation_current,
                            agent_action,
                            reward: float,
                            agent_observation_next,
                            train_step_current: int, train_step_absolute: int,
                            train_episode_current: int, train_episode_absolute: int,
                            train_episode_volley: int, train_episode_total: int):
        # Save the current step in the buffer together with the current value estimate
        self._model.buffer.store(agent_observation_current, agent_action, reward, self._current_value_estimate, self._current_log_likelihood)

    def complete_step_inference(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
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
                                interface: Interface,
                                last_step_reward: float,
                                episode_total_reward: float,
                                warmup_episode_current: int,
                                warmup_steps_volley: int):
        pass

    def complete_episode_train(self,
                               logger: logging.Logger,
                               session,
                               interface: Interface,
                               last_step_reward: float,
                               episode_total_reward: float,
                               train_step_absolute: int,
                               train_episode_current: int, train_episode_absolute: int,
                               train_episode_volley: int, train_episode_total: int):
        # Update the buffer at the end of trajectory
        self._model.buffer.finish_path(last_step_reward)
        # Increase trade-off by its update value
        self._dirichlet_trade_off = min(self._dirichlet_trade_off_max, self._dirichlet_trade_off + self._dirichlet_trade_off_update)
        # Execute update only after a certain number of trajectories each time (need to offset the episode by 1 to prevent not updating last episode)
        if (train_episode_current + 1) % (train_episode_volley / self.updates_per_training_volley) == 0 and train_episode_current > 0:
            # Execute the update and store policy and value loss
            buffer: [] = self._model.buffer.get()
            logger.info("Updating the model with a buffer of " + str(len(buffer[0])) + " samples")
            summary, self._current_policy_loss, self._current_value_loss = self._model.update(session, buffer)
            # Update the summary at the absolute current step
            self._summary_writer.add_summary(summary, train_step_absolute)

    def complete_episode_inference(self,
                                   logger: logging.Logger,
                                   session,
                                   interface: Interface,
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
    def warmup_steps(self) -> int:
        # Return the amount of warmup episodes required by the model
        return self._model.warmup_steps



