# Import packages

import tensorflow
import numpy
import time
import logging

# Import required src

from usienarl.environment import Environment, SpaceType


class Visualizer:
    """
    Visualizer of models. It uses a saved model in the form of a metagraph file and run a specified number of episodes
    on a specified environment.

    It is used to further test a saved model and to visualize, if the render option is selected, the behaviour of the
    model in full inference mode.
    """

    def __init__(self,
                 environment: Environment,
                 model_name: str, inputs_name: str, outputs_name: str,
                 metagraph_directory_path: str, metagraph_name: str,
                 logger: logging.Logger):
        # Define the logger
        self._logger: logging.Logger = logger
        # Define the environment
        self._environment: Environment = environment
        # Define tested model variables
        self._model_name: str = model_name
        self._inputs_name: str = inputs_name
        self._outputs_name: str = outputs_name
        self._metagraph_directory_path: str = metagraph_directory_path
        self._metagraph_name: str = metagraph_name
        self._outputs = None
        self._inputs = None

    def _get_metagraph_prediction(self,
                                  session,
                                  state_current):
        """
        Get the prediction from the metagraph given the current state.

        :param session: the session of tensorflow currently running
        :param state_current: the current state in the environment
        :return: the best predicted action
        """
        # Return all the predicted actions q-values given the current state depending on the observation space type
        if self._environment.observation_space_type is SpaceType.discrete:
            predicted_outputs = session.run(self._outputs,
                                            feed_dict={self._inputs: [numpy.identity(self._environment.observation_space_shape)[state_current]]})
        else:
            predicted_outputs = session.run(self._outputs,
                                            feed_dict={self._inputs: [state_current]})
        # Return the best predicted action (the index of the best predicted output)
        return numpy.argmax(predicted_outputs)

    def run(self,
            episodes: int,
            render: bool = False, frame_rate: int = 0):
        """
        Run the visualizer for the given number of episodes. If it should render, a frame rate should be specified.

        :param episodes: the number of episodes for which to visualize the model
        :param render: a boolean flag specifying whether or not the environment step should be rendered
        :param frame_rate: an integer indicating how many frame per seconds of rendering should be shown (it has to be >= 0)
        """
        # Reset the tensorflow default graph
        tensorflow.reset_default_graph()
        # Setup the environment
        self._environment.setup()
        # Import the metagraph of the experiment
        model_saver = tensorflow.train.import_meta_graph(self._metagraph_directory_path + "/" + self._metagraph_name + ".meta")
        with tensorflow.Session() as session:
            # Initialize the environment
            self._environment.initialize(session)
            # Load the last checkpoint of the defined metagraph
            model_saver.restore(session, tensorflow.train.latest_checkpoint(self._metagraph_directory_path))
            # Get the default graphs and set the inputs and the outputs of the model
            graph = tensorflow.get_default_graph()
            self._inputs = graph.get_tensor_by_name(self._metagraph_name + "/" + self._model_name + "/" + self._inputs_name + ":0")
            self._outputs = graph.get_tensor_by_name(self._metagraph_name + "/" + self._model_name + "/" + self._outputs_name + ":0")
            # Start visualizing
            self._logger.info("Visualize for " + str(episodes) + " episodes...")
            # Define list of rewards
            rewards = []
            for episode in range(episodes):
                self._logger.info("Visualizing episode " + str(episode))
                # Initialize reward and episode completion flag
                episode_reward: float = 0
                episode_done: bool = False
                # Get the initial state of the episode
                state_current = self._environment.reset(session)
                # Execute actions until the episode is completed
                while not episode_done:
                    # Get the action predicted by the currently loaded metagraph
                    action: int = self._get_metagraph_prediction(session, state_current)
                    # Render the environment if required
                    if render:
                        self._environment.render(session)
                    # Get the next state with relative reward and completion flag
                    state_next, reward, episode_done, _ = self._environment.step(action, session)
                    # Update total reward for this episode
                    episode_reward += reward
                    # Update the current state with the previously next state
                    state_current = state_next
                    # Wait for the next frame
                    if frame_rate > 0:
                        frame_time: float = 1.0 / frame_rate
                        time.sleep(frame_time)
                # Add the episode reward to the list of rewards
                rewards.append(episode_reward)
                # Return average reward over given episodes
                self._logger.info("Average reward over " + str(episodes) + " is " + str(sum(rewards) / episodes))
            self._environment.close(session)
