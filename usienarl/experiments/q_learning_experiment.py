# Import packages

import logging

# Import required src

from usienarl import Experiment, Environment, Explorer, Memory
from usienarl.models import TemporalDifferenceModel


class QLearningExperiment(Experiment):
    """
    Experiment in which the model commanding the agent is a Q-Learning algorithm.
    It is the same as any other experiment, but the training system is tuned for Q-Learning.

    It can only accept Q-Learning models.
    """

    def __init__(self,
                 name: str,
                 validation_success_threshold: float, test_success_threshold: float,
                 environment: Environment,
                 model: TemporalDifferenceModel,
                 memory: Memory, batch_size: int,
                 explorer: Explorer):
        # Define temporal_difference experiment attributes
        # Define the memory: set also the number of pre-training episodes depending on the memory requirements and capacity
        self._memory: Memory = memory
        self._batch_size: int = batch_size
        pre_training_episodes: int = 0
        if self._memory is not None and self._memory.pre_train:
            pre_training_episodes = self._memory.capacity
        # Define the explorer and initialize the exploration rate
        self._explorer: Explorer = explorer
        self._exploration_rate: float = self._explorer.exploration_rate_start_value
        super().__init__(name, validation_success_threshold, test_success_threshold, environment, model, pre_training_episodes)

    def _reset(self):
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        # Reset memory and explorer, if defined
        if self._explorer is not None:
            self._exploration_rate = self._explorer.exploration_rate_start_value
        if self._memory is not None:
            self._memory.reset()

    def _pre_train(self,
                   episodes: int, session,
                   logger: logging.Logger):
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        # Initialize the pre-training process
        logger.info("Pre-training for " + str(episodes) + " episodes...")
        for episode in range(episodes):
            # Initialize episode completion flag
            episode_done: bool = False
            # Get the initial state of the episode
            state_current = self.environment.reset(session)
            # Execute actions (one per step) until the episode is completed
            while not episode_done:
                # Take a random action
                action: int = self.environment.get_random_action(session)
                # Get the next state with relative reward and completion flag
                state_next, reward, episode_done = self.environment.step(action, session)
                # For storage purposes, set the state next to none if the episode is completed
                if episode_done:
                    state_next = None
                # Add the new acquired sample in the memory
                self._memory.add_sample((state_current, action, reward, state_next))
                # Update the current state with the previously next state
                state_current = state_next
            # Print intermediate pre-train completion (every 1/10 of the pre-training episodes)
            if episode % (episodes // 10) == 0 and episode > 0:
                logger.info("Pre-trained for " + str(episode) + " episodes")

    def _train(self,
               experiment_number: int,
               summary_writer, model_saver, save_path: str,
               episodes: int, start_step: int, session,
               logger: logging.Logger,
               render: bool = False):
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        # Initialize the training process
        self.model.train_initialize(session)
        logger.info("Training for " + str(episodes) + " episodes...")
        # Count the steps of all episodes
        step: int = 0
        for episode in range(episodes):
            # Initialize reward and episode completion flag
            episode_reward: float = 0
            episode_done: bool = False
            # Get the initial state of the episode
            state_current = self.environment.reset(session)
            # Update the exploration rate at each episode except the first
            if episode > 0:
                self._exploration_rate = self._explorer.update_exploration_rate(self._exploration_rate)
            # Execute actions (one per step) until the episode is completed
            while not episode_done:
                # Increment the step counter
                step += 1
                # Get the action predicted by the model according to the explorer strategy
                action: int = self._explorer.get_action(self._exploration_rate, self.model, self.environment, session, state_current)[0]
                # Get the next state with relative reward and completion flag
                state_next, reward, episode_done = self.environment.step(action, session)
                # For storage purposes, set the state next to none if the episode is completed
                if episode_done:
                    state_next = None
                # Update the model with batch update or single update depending on memory (if defined, use batch update)
                if self._memory is not None:
                    # Add the new acquired sample in the memory
                    self._memory.add_sample((state_current, action, reward, state_next))
                    # Get the samples with related weights
                    samples, samples_weights = self._memory.get_sample(self._batch_size)
                    # Get the loss and the absolute error for each target/prediction at this step
                    loss, absolute_error, summary = self.model.update_batch(session,
                                                                            episode, episodes, step,
                                                                            samples, samples_weights)
                    # Update the memory by using the absolute error
                    self._memory.update(absolute_error)
                else:
                    # Get the loss and the absolute error for each target/prediction at this step
                    loss, absolute_error, summary = self.model.update_single(session,
                                                                             episode, episodes, step,
                                                                             state_current, state_next, action, reward)
                # Update total reward for this episode
                episode_reward += reward
                # Update the current state with the previously next state
                state_current = state_next
                # Update the summary writer
                summary_writer.add_summary(summary, step + start_step)
                # Render if required
                if render:
                    self.environment.render(session)
        # Save the model
        logger.info("Saving the model...")
        if experiment_number >= 0:
            model_saver.save(session, save_path + "/" + self.name + "_" + str(experiment_number))
        else:
            model_saver.save(session, save_path + "/" + self.name)
        # Return the reached step
        return step + start_step
