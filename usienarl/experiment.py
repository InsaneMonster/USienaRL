# Import packages

import tensorflow
import numpy
import logging

# Import required src

from usienarl import Environment, Agent, Interface


class Experiment:
    """
    TODO: summary
    """

    def __init__(self,
                 name: str,
                 environment: Environment,
                 agent: Agent,
                 interface: Interface = None):
        # Define the experiment name
        self.name: str = name
        # Define environment, agent and interface attributes
        self.environment: Environment = environment
        self.agent: Agent = agent
        self.interface: Interface = interface
        # Define experiment internal attributes
        self._agent_saver = None
        self._metagraph_path: str = None
        self._tensorflow_gpu_config = None
        self._current_id: str = None

    def setup(self,
              summary_path: str, metagraph_path: str,
              logger: logging.Logger,
              iteration: int = -1) -> bool:
        """
        Setup the experiment, preparing all of its component to execution. This must be called before conduct method.

        :param summary_path: the string path of the TensorBoard summary directory to save during model training
        :param metagraph_path: the string path of the saved model directory to save at the end of each training interval
        :param logger: the logger used to print the experiment information, warnings and errors
        :param iteration: number to append to the experiment name in all scopes and print statements (if not less than zero)
        :return: a boolean equals to True if the setup of the experiment is successful, False otherwise
        """
        # Save current experiment id using name and iteration
        self._current_id = self.name + "_" + str(iteration) if iteration > -1 else self.name
        # Start setup process
        logger.info("Setup of experiment " + self._current_id + "...")
        # Reset the tensorflow graph to a blank new graph
        tensorflow.reset_default_graph()
        # Execute the setup on the environment and on the agent and return if not both of them setup correctly
        logger.info("Environment setup..")
        if not self.environment.setup(logger):
            logger.info("Environment setup failed. Cannot setup the experiment!")
            return False
        logger.info("Environment setup is successful!")
        logger.info("Agent setup..")
        if not self.agent.setup(logger, self._current_id, summary_path):
            logger.info("Agent setup failed. Cannot setup the experiment!")
            return False
        logger.info("Agent setup is successful!")
        # If setup of both environment and agent is successful, reset internal experiment variables
        self._metagraph_path = metagraph_path
        # Initialize the agent internal model saver
        self._agent_saver = tensorflow.train.Saver(self.agent.trainable_variables)
        logger.info("Agent internal model will be saved after each train volley")
        logger.info("Agent internal model metagraph save path: " + self._metagraph_path)
        # Initialize tensorflow gpu configuration
        self._tensorflow_gpu_config = tensorflow.ConfigProto()
        self._tensorflow_gpu_config.gpu_options.allow_growth = True
        logger.info("CUDA GPU device configured for Tensorflow-gpu")
        # Setup is successful
        logger.info("Experiment setup is successful. Now can conduct the experiment")
        return True

    def _pre_train(self,
                   logger: logging.Logger,
                   episodes: int, session,
                   render: bool = False):
        """
        Execute a volley of pre-training of the agent on the environment.
        How the pre-training is done depends on the experiment implementation.

        Note: to call a pre-train step, use the appropriate private method.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param episodes: the number of episodes in which to run the agent in pre-train mode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during pre-training
        :return: the float average of the score obtained in the played episodes
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def _train(self,
               logger: logging.Logger,
               episodes: int, session,
               render: bool = False):
        """
        Execute a volley of training of the agent on the environment.
        How the training is done depends on the experiment implementation.

        Note: to call a train step, use the appropriate private method.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param episodes: the number of episodes in which to run the agent in train mode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during training
        :return: the float average of the score obtained in the played episodes
        """
        # Define list of scores
        scores: numpy.ndarray = numpy.zeros(episodes, dtype=float)
        for episode in range(episodes):
            # Initialize score and episode completion flag
            episode_score: float = 0
            episode_done: bool = False
            # Get the initial state of the episode
            state_current = self.environment.reset(logger, session)
            # Initialize the step of the episode
            step: int = 0
            # Execute actions until the episode is completed or the maximum length is exceeded
            while not episode_done and not step > self._episode_length_max:
                # Increment the step counter
                step += 1
                # Execute a train step
                # Get the action decided by the agent with pre-train policy
                action = self.agent.act_train(logger, session, state_current)
                # Get the next state with relative reward and completion flag
                state_next, reward, episode_done = self.environment.step(logger, action, session)
                # Render if required
                if render:
                    self.environment.render(logger, session)
                # Finish the step and send back information to the agent
                self.agent.finish_step_train(logger, session, state_current, action, reward, state_next)
                # Update score for this episode with the given reward
                episode_score += reward
                # Update the current state with the previously next state
                state_current = state_next
            # Finish the episode and send back information to the agent
            self.agent.finish_episode_train(logger, session, episode_reward)
            # Add the episode reward to the list of rewards
            scores[episode] = episode_score
        # Return back the scores array to the train method
        return scores

    def _inference(self,
                   logger: logging.Logger,
                   episodes: int, session,
                   render: bool = False):
        """
        Execute a volley of inference of the agent on the environment.
        How the inference is done depends on the experiment implementation.

        Note: to call an inference step, use the appropriate private method.

        :param logger: the logger used to print the experiment information, warnings and errors
        :param episodes: the number of episodes in which to run the agent in inference mode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during inference
        :return: the float average of the score obtained in the played episodes
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def conduct(self,
                training_episodes_per_volley: int, validation_episodes_per_volley: int, max_training_episodes: int,
                test_episodes_per_cycle: int, test_cycles: int,
                logger: logging.Logger,
                render_during_training: bool = False, render_during_validation: bool = False, render_during_test: bool = False):
        """
        Conduct the experiment of the given number of training (up to the given maximum) and validation episodes
        per volley, with the associated given number of test episodes per cycle and the given number of test cycles after
        the validation has passed.

        A volley of training is always considered to stop the training exactly when the model performs well enough
        to validate, by checking the appropriate condition. When validation is successful, appropriate condition is also
        checked on the test cycles to decide if the experiment is successful or not.

        Conducting an experiment will generate a tensorflow session for that experiment. It is required that the experiment
        is ready to be conducted by checking the result of the setup method before running this one.

        :param training_episodes_per_volley: the number of training episodes per volley before trying to validate the model
        :param validation_episodes_per_volley: the number of validation episodes per volley after the training in such interval
        :param max_training_episodes: the maximum number of training episodes allowed at all
        :param test_episodes_per_cycle: the number of episodes to play for each test cycle after validation has passed
        :param test_cycles: the number of test cycles to execute after validation has passed
        :param logger: the logger used to print the experiment information, warnings and errors
        :param render_during_training: boolean flag to render the environment during training
        :param render_during_validation: boolean flag to render the environment during validation
        :param render_during_test: boolean flag to render the environment during test
        :return: the average of averages score over all the test cycles, the best average score among all the test cycles and the training episodes required to validate the model
        """
        logger.info("Conducting experiment " + self._current_id + "...")
        # Define the session
        with tensorflow.Session(config=self._tensorflow_gpu_config) as session:
            # Initialize the environment and the agent
            self.environment.initialize(logger, session)
            self.agent.initialize(logger, session)
            # Execute pre-training if the agent requires pre-training
            if self.agent.pre_train_episodes > 0:
                logger.info("Pre-training for " + str(self.agent.pre_train_episodes) + " episodes...")
                self._pre_train(logger, self.agent.pre_train_episodes, session)
            # Execute training until max training episodes number is reached or the validation score is above the threshold
            while self._train_episodes_counter < max_training_episodes:
                # Run train for training episodes per volley and get the average score
                logger.info("Training for " + str(training_episodes_per_volley) + " episodes...")
                training_score: float = self._train(logger, training_episodes_per_volley, render_during_training)
                # Print training results
                logger.info("Training of " + str(training_episodes_per_volley) + " finished with following result:")
                logger.info("Average score over " + str(training_episodes_per_volley) + " training episodes after " + str(self._train_episodes_counter) + " total training episodes: " + str(training_score))
                logger.info("Total training steps: " + str(self.agent.train_steps_counter))
                # Increase training episodes counter
                self._train_episodes_counter += training_episodes_per_volley
                # Save the agent internal model at the current step
                logger.info("Saving the model...")
                self._agent_saver.save(session, self._metagraph_path + "/" + self._current_id)
                # Run inference for validation episodes per volley and get the average score
                logger.info("Validating for " + str(validation_episodes_per_volley) + " episodes...")
                validation_score: float = self._inference(logger, validation_episodes_per_volley, session, render_during_validation)
                # Print validation results
                logger.info("Training of " + str(validation_episodes_per_volley) + " finished with following result:")
                logger.info("Average score over " + str(validation_episodes_per_volley) + " validation episodes after " + str(self._train_episodes_counter) + " total training episodes: " + str(validation_score))
                # Check for validation
                if self._is_validated(self._train_episodes_counter, self.agent.train_steps_counter, validation_score, training_score):
                    logger.info("Validation of the model is successful")
                    break
            # Test the model, getting the average score per test and an array of all tests
            test_scores: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
            for test_index in range(test_cycles):
                # Run inference for test episodes per cycles and get the average score
                logger.info("Testing for " + str(test_episodes_per_cycle) + " episodes...")
                score_cycle: float = self._inference(logger, test_episodes_per_cycle, session, render_during_test)
                # Print validation results
                logger.info("Testing of " + str(test_episodes_per_cycle) + " finished with following result:")
                logger.info("Average score over " + str(test_episodes_per_cycle) + " test episodes: " + str(score_cycle))
                # Insert the score to the array of scores
                test_scores[test_index] = score_cycle
            # Get the final average score over all the test and the best score registered among all tests
            average_test_score: float = numpy.average(test_scores)
            best_test_score: float = numpy.max(test_scores)
            # Print final results and outcome of the experiment
            logger.info("Average test score is " + str(average_test_score) + " with " + str(self._train_episodes_counter) + " training episodes")
            logger.info("Best average test score over " + str(test_episodes_per_cycle) + " is: " + str(best_test_score))
            # Check if the experiment is successful
            if self._is_successful(self._train_episodes_counter, self.agent.train_steps_counter, average_test_score, best_test_score):
                logger.info("The experiment is successful")
            else:
                logger.info("The experiment is not successful")
            return average_test_score, best_test_score, self._train_episodes_counter

    def _is_validated(self,
                      total_training_episodes: int, total_training_steps: int,
                      average_validation_score: float, average_training_score: float) -> bool:
        """
        Check if the experiment is to be considered validated using the given parameters.

        :param total_training_episodes: the total number of training episodes up to the validation time
        :param total_training_steps: the total number of training steps up to the validation time
        :param average_validation_score: the average validation score in the last volley
        :param average_training_score: the average training score in the last volley
        :return: a boolean flag True if condition are satisfied, False otherwise
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def _is_successful(self,
                       total_training_episodes: int, total_training_steps: int,
                       average_test_score: float, best_test_score: float) -> bool:
        """
        Check if the experiment is to be considered successful using the given parameters.

        :param total_training_episodes: the total number of training episodes up to the test time
        :param total_training_steps: the total number of training steps up to the test time
        :param average_test_score: the average of averages of testing scores among all test cycles
        :param best_test_score: the best of averages of testing scores among all test cycles
        :return: a boolean flag True if condition are satisfied, False otherwise
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()
