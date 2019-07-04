# Import packages

import tensorflow
import numpy
import logging
import time

# Import required src

from usienarl import Environment, Agent
from usienarl import Model


class Experiment:
    """
    Experiment with a certain model/parameters on a certain environment in which the agent guided by the model operates.
    It contains all data and methods to perform any kind of experiments.

    The standard experiment run is the following:
    - setup of the experiment
    - train for an interval of episodes (in training mode, also executes pre-training if necessary)
    - validate for a certain number of episodes (in exploit mode)
    - if validation is successful run a certain number of tests, otherwise train for another interval
    - run all the tests, each one for the required number of episodes (in exploit mode)

    The training mode updates the model parameters, the exploit mode just run the model to get the prediction on the
    best next action to perform by the agent on the environment.

    Any experiment requires a name for the experiments itself (usually after its environment), a validation and success threshold.
    The validation threshold is checked after any training interval, to check if it the proper time to exit training and start testing.
    The success threshold is checked during testing to determine whether or not the experiment is successful.

    An experiment can run only on one environment and with only one combination of model, memory and explorer.
    The batch size is required when specifying the memory (when the memory is not None) in order to choose how many
    samples to use when updating the model.

    Use the method conduct to run the experiment first in training and, after validation, in test mode.
    Use the method setup to prepare the environment for the next run.

    Attributes:
        - name: a string representing the name of the experiment, usually correlated with the name of the environment
        - environment: an environment representing the setting of the experiment (e.g. an OpenAI gym environment)
        - model: the temporal_difference model moving the agent operating on the environment
    """

    def __init__(self,
                 name: str,
                 validation_success_threshold: float, test_success_threshold: float,
                 environment: Environment,
                 agent: Agent,
                 pre_train_episodes: int):
        # Define the experiment name
        self.name: str = name
        # Define environment and agent attributes
        self.environment: Environment = environment
        self.agent: Agent = agent
        # Define experiment internal attributes
        self._validation_success_threshold: float = validation_success_threshold
        self._test_success_threshold: float = test_success_threshold
        self._pre_train_episodes: int = pre_train_episodes
        ####
        self._agent_saver = None
        self._summary_writer = None
        self._summary_path: str = None
        self._metagraph_path: str = None
        self._tensorflow_gpu_config = None
        self._train_steps_counter: int = 0
        self._train_episodes_counter: int = 0

    def setup(self,
              summary_path: str, metagraph_path: str,
              logger: logging.Logger,
              iteration: int = -1) -> bool:
        """
        Setup the experiment, preparing all of its component to execution. This must be called before conduct method.

        :param summary_path: the string path of the TensorBoard summary directory to save during model training
        :param metagraph_path: the string path of the saved model directory to save at the end of each training interval
        :param logger: the logger currently used to print the experiment information, warnings and errors
        :param iteration: number to append to the experiment name in all scopes and print statements (if not less than zero)
        :return: a boolean equals to True if the setup of the experiment is successful, False otherwise
        """
        logger.info("Setup of experiment " + self.name + "_" + str(iteration) + "...")
        # Reset the tensorflow graph to a blank new graph
        tensorflow.reset_default_graph()
        # Execute the setup on the environment and on the agent and return if not both of them setup correctly
        logger.info("Environment setup..")
        if not self.environment.setup():
            logger.info("Environment setup failed. Cannot setup the experiment!")
            return False
        logger.info("Environment setup is successful!")
        logger.info("Agent setup..")
        if not self.agent.setup():
            logger.info("Agent setup failed. Cannot setup the experiment!")
            return False
        logger.info("Agent setup is successful!")
        # If setup of both environment and agent is successful, reset internal experiment variables
        self._train_steps_counter = 0
        self._train_episodes_counter = 0
        self._summary_path = summary_path
        self._metagraph_path = metagraph_path
        # Configure the tensorboard summary and the agent saver
        logger.info("Tensorboard summary path: " + self._summary_path)
        logger.info("Agent internal model metagraph save path: " + self._metagraph_path)
        # Initialize the tensorboard summary writer and the agent internal model saver
        self._agent_saver = tensorflow.train.Saver(self.agent.get_trainable_variables(self.name))
        self._summary_writer = tensorflow.summary.FileWriter(self._summary_path, graph=tensorflow.get_default_graph())
        # Initialize tensorflow gpu configuration
        self._tensorflow_gpu_config = tensorflow.ConfigProto()
        self._tensorflow_gpu_config.gpu_options.allow_growth = True
        # Setup is successful
        logger.info("Experiment setup is successful. Now can conduct the experiment")
        return True

        # Generate the model and return the success/failure state of this generation
        # return self.model.generate(self.name + "_" + str(experiment_number),
        #                            self.environment.observation_space_type, self.environment.observation_space_shape,
        #                            self.environment.action_space_type, self.environment.action_space_shape,
        #                            logger)

    def _pre_train(self,
                   episodes: int, session,
                   render: bool = False):
        """
        Execute a volley of pre-training of the agent on the environment.
        How the pre-training is done depends on the experiment implementation.

        Note: to call a pre-train step, use the appropriate private method.

        :param episodes: the number of episodes in which to run the agent in pre-train mode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during pre-training
        :return: the float average of the score obtained in the played episodes
        """
        # Empty method, definition should be implemented on a child class basis
        pass

    def _train(self,
               episodes: int, session,
               render: bool = False):
        """
        Execute a volley of training of the agent on the environment.
        How the training is done depends on the experiment implementation.

        Note: to call a train step, use the appropriate private method.

        :param episodes: the number of episodes in which to run the agent in train mode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during training
        :return: the float average of the score obtained in the played episodes
        """
        # Empty method, definition should be implemented on a child class basis
        pass

    def _inference(self,
                   episodes: int, session,
                   render: bool = False):
        """
        Execute a volley of inference of the agent on the environment.
        How the inference is done depends on the experiment implementation.

        Note: to call an inference step, use the appropriate private method.

        :param episodes: the number of episodes in which to run the agent in inference mode
        :param session: the session of tensorflow currently running
        :param render: boolean parameter deciding whether or not to render during inference
        :return: the float average of the score obtained in the played episodes
        """
        # Empty method, definition should be implemented on a child class basis
        pass

    def _pre_train_step(self,
                        session,
                        state_current: numpy.ndarray,
                        render: bool = False):
        """
        Execute a pre-train step in the experiment.

        :param session: the session of tensorflow currently running
        :param state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :param render: a boolean flag stating if the environment should be rendered in the step
        :return: the next state, the reward and the completion flag
        """
        # Get the action decided by the agent with pre-train policy
        action = self.agent.act_pre_train(session, state_current)
        # Get the next state with relative reward and completion flag
        state_next, reward, episode_done = self.environment.step(action, session)
        # Render if required
        if render:
            self.environment.render(session)
        # Return the next state, the reward and the completion flag
        return state_next, reward, episode_done

    def _train_step(self,
                    session,
                    state_current: numpy.ndarray,
                    render: bool = False):
        """
        Execute a train step in the environment.

        :param session: the session of tensorflow currently running
        :param state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :param render: a boolean flag stating if the environment should be rendered in the step
        :return: the next state, the reward and the completion flag
        """
        # Increase the train steps counter
        self._train_steps_counter += 1
        # Get the action decided by the agent with train policy
        action = self.agent.act_pre_train(session, state_current)
        # Get the next state with relative reward and completion flag
        state_next, reward, episode_done = self.environment.step(action, session)
        # Render if required
        if render:
            self.environment.render(session)
        # Return the next state, the reward and the completion flag
        return state_next, reward, episode_done

    def _inference_step(self,
                        session,
                        state_current: numpy.ndarray,
                        render: bool = False):
        """
        Execute an inference step in the environment.

        :param session: the session of tensorflow currently running
        :param state_current: the current state of the environment wrapped in a numpy array (ndarray)
        :param render: a boolean flag stating if the environment should be rendered in the step
        :return: the next state, the reward and the completion flag
        """
        # Get the action decided by the agent with inference policy
        action = self.agent.act_inference(session, state_current)
        # Get the next state with relative reward and completion flag
        state_next, reward, episode_done = self.environment.step(action, session)
        # Render if required
        if render:
            self.environment.render(session)
        # Return the next state, the reward and the completion flag
        return state_next, reward, episode_done

    def conduct(self,
                training_episodes_per_volley: int, validation_episodes_per_volley: int, max_training_episodes: int,
                test_episodes_per_cycle: int, test_cycles: int,
                logger: logging.Logger,
                render_during_training: bool = False, render_during_validation: bool = False, render_during_test: bool = False,
                experiment_number: int = -1):
        """
        Conduct the experiment of the given number of training (up to the given maximum) and validation episodes
        per interval, with the associated given number of episodes per test and the given number of tests after
        the validation has passed.

        An interval of training is always considered to stop the training exactly when the model performs well enough
        to validate, having an average score greater than the required validation threshold.

        Conducting an experiment will generate a tensorflow session for that experiment. The tensorflow graph is reset
        and the default one is substituted with the one required by the ongoing experiment.

        :param training_episodes_per_volley: the number of training episodes per interval before trying to validate the model
        :param validation_episodes_per_volley: the number of validation episodes per interval after the training in such interval
        :param max_training_episodes: the maximum number of training episodes allowed in one interval
        :param test_episodes_per_cycle: the number of episodes to play for each test after validation has passed
        :param test_cycles: the number of tests to execute after validation has passed
        :param logger: the logger currently used to print the experiment information, warnings and errors
        :param render_during_training: boolean flag to render the environment during training (with automatic frame rate)
        :param render_during_validation: boolean flag to render the environment during validation (with automatic frame rate)
        :param render_during_test: boolean flag to render the environment during test (with automatic frame rate)
        :param experiment_number: number to append to the experiment name in all scopes and print statements (if not less than zero). Used to differentiate multiple copies of the same experiment
        :return: the final average score over all the tests, the best average score in a test among all the test and the training episodes required to validate the model
        """
        if experiment_number >= 0:
            logger.info("Conducting experiment " + self.name + "_" + str(experiment_number) + "...")
        else:
            logger.info("Conducting experiment " + self.name + "...")
        # Define the session
        with tensorflow.Session(config=self._tensorflow_gpu_config) as session:
            # Initialize the environment and the agent
            self.environment.initialize(session)
            self.agent.initialize(session)
            # Execute pre-training if the agent requires pre-training
            if self.agent.require_pre_train:
                logger.info("Pre-training for " + str(self._pre_train_episodes) + " episodes...")
                self._pre_train(self._pre_train_episodes, session)
            # Execute training until max training episodes number is reached or the validation score is above the threshold
            while self._train_episodes_counter < max_training_episodes:
                # Run train for training episodes per volley and get the average score
                # Also compute the time required for train
                logger.info("Training for " + str(training_episodes_per_volley) + " episodes...")
                start_time: int = int(round(time.time() * 1000))
                training_score: float = self._train(training_episodes_per_volley, render_during_training)
                end_time: int = int(round(time.time() * 1000))
                # Print training results
                logger.info("Training of " + str(training_episodes_per_volley) + " finished in " + str(end_time - start_time) + " msec with following result:")
                logger.info("Average score over " + str(training_episodes_per_volley) + " training episodes after " + str(self._train_episodes_counter) + " total training episodes: " + str(training_score))
                logger.info("Total training steps: " + str(self._train_steps_counter))
                # Increase training episodes counter
                self._train_episodes_counter += training_episodes_per_volley
                # Save the agent internal model at the current step
                logger.info("Saving the model...")
                self._agent_saver.save(session, self._metagraph_path + "/" + self.name)
                # Run inference for validation episodes per volley and get the average score
                # Also compute the time required for validation
                logger.info("Validating for " + str(validation_episodes_per_volley) + " episodes...")
                start_time: int = int(round(time.time() * 1000))
                validation_score: float = self._inference(validation_episodes_per_volley, session, render_during_validation)
                end_time: int = int(round(time.time() * 1000))
                # Print validation results
                logger.info("Training of " + str(validation_episodes_per_volley) + " finished in " + str(end_time - start_time) + " msec with following result:")
                logger.info("Average score over " + str(validation_episodes_per_volley) + " validation episodes after " + str(self._train_episodes_counter) + " total training episodes: " + str(validation_score))
                # Check for validation
                if self._is_validated(self._train_episodes_counter, self._train_steps_counter, validation_score, training_score):
                    logger.info("Validation of the model is successful")
                    break
            # Test the model, getting the average score per test and an array of all tests
            test_scores: numpy.ndarray = numpy.zeros(test_cycles, dtype=float)
            for test_index in range(test_cycles):
                # Run inference for test episodes per cycles and get the average score
                # Also compute the time required for test
                logger.info("Testing for " + str(test_episodes_per_cycle) + " episodes...")
                start_time: int = int(round(time.time() * 1000))
                score_cycle: float = self._inference(test_episodes_per_cycle, session, render_during_test)
                end_time: int = int(round(time.time() * 1000))
                # Print validation results
                logger.info("Testing of " + str(test_episodes_per_cycle) + " finished in " + str(end_time - start_time) + " msec with following result:")
                logger.info("Average score over " + str(test_episodes_per_cycle) + " test episodes: " + str(score_cycle))
                # Insert the score to the array of scores
                test_scores[test_index] = score_cycle
            # Get the final average score over all the test and the best score registered among all tests
            average_test_score: float = numpy.average(test_scores)
            best_test_score: float = numpy.max(test_scores)
            # Print final results and outcome of the experiment
            logger.info("Average test score is " + str(average_test_score) + " with " + str(self._train_episodes_counter) + " training episodes")
            logger.info("Best average test score over " + str(test_episodes_per_cycle) + " is: " + str(best_test_score))
            if self._is_successful(self._train_episodes_counter, self._train_steps_counter, average_test_score, best_test_score):
                logger.info("The experiment is successful")
            else:
                logger.info("The experiment is not successful")
            return average_test_score, best_test_score, self._train_episodes_counter

    def _is_validated(self,
                      total_training_episodes: int, total_training_steps: int,
                      average_validation_score: float, average_training_score: float) -> bool:
        """


        :param total_training_episodes:
        :param total_training_steps:
        :param average_validation_score:
        :param average_training_score:
        :return:
        """
        pass

    def _is_successful(self,
                       total_training_episodes: int, total_training_steps: int,
                       average_test_score: float, best_test_score: float) -> bool:
        """


        :param total_training_episodes:
        :param total_training_steps:
        :param average_test_score:
        :param best_test_score:
        :return:
        """
        pass
