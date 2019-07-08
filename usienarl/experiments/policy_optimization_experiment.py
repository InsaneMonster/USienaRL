# Import packages

import logging

# Import required src

from usienarl.experiment import Experiment, Environment
from usienarl.models import PolicyOptimizationModel


class PolicyOptimizationExperiment(Experiment):
    """
    Experiment in which the model commanding the agent is a Policy Optimization algorithm.
    It is the same as any other experiment, but the training system is tuned for Policy Optimization.

    It can only accept Policy Optimization models.
    """

    def __init__(self,
                 name: str,
                 validation_success_threshold: float, test_success_threshold: float,
                 environment: Environment,
                 model: PolicyOptimizationModel,
                 updates_per_training_interval: int):
        # Define policy optimization experiment attributes
        self._updates_per_training_interval: int = updates_per_training_interval
        # Generate the base experiment
        super().__init__(name, validation_success_threshold, test_success_threshold, environment, model, 0)

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
        logger.info("Training for " + str(episodes) + " episodes...")
        # Count the steps of all episodes and the number of updates
        step: int = 0
        update_count: int = 0
        for episode in range(episodes):
            # Initialize reward and episode completion flag
            episode_reward: float = 0
            episode_done: bool = False
            # Get the initial state of the episode
            state_current = self._environment.reset(session)
            # Execute actions (one per step) until the episode is completed
            while not episode_done:
                # Increment the step counter
                step += 1
                # Get the action predicted by the model on the current state with the current policy and also the estimated value
                result: [] = self.model.predict(session, state_current)
                action: int = result[0]
                # Get the next state with relative reward and completion flag
                state_next, reward, episode_done = self._environment.step(action, session)
                # Update total reward for this episode
                episode_reward += reward
                # Store the properties in the model buffer (current state, action and relative reward then all the content
                # returned by the model during prediction, since they can vary depending on the algorithm)
                self.model.buffer.store_train(state_current, action, reward, *result[1:])
                # If the episode is completed and finalize the path in the model buffer
                # Else update the current state with the previously next state
                if episode_done:
                    self.model.buffer.finish_path(reward)
                else:
                    state_current = state_next
                # Render if required
                if render:
                    self._environment.render(session)
            # Update the model each defined steps
            if episode % (episodes / self._updates_per_training_interval) == 0 and episode > 0:
                # Increase the update count
                update_count += 1
                # Get the losses of the policy for target/prediction for the observed trajectories in the batch and the
                # correlated summary (the batch is stored inside a buffer in the model class)
                policy_loss, value_loss, summary = self.model.update(session, self.model.buffer.get())
                # Update the summary writer
                summary_writer.add_summary(summary, step + start_step)
        # Save the model
        logger.info("Executed " + str(update_count) + " updates count in " + str(step) + " steps")
        logger.info("Saving the model...")
        if experiment_number >= 0:
            model_saver.save(session, save_path + "/" + self._name + "_" + str(experiment_number))
        else:
            model_saver.save(session, save_path + "/" + self._name)
        # Return the reached step
        return step + start_step

