# Import packages

import numpy
import tensorflow
import logging

# Import required src

from usienarl import SpaceType


class Model:
    """
    Base class for each model. It should not be used by itself, and should be extended instead.

    When creating an instance of a model (of a subclass of it) the model is not yet defined.
    To define a proper tensorflow graph additional data regarding the experiment should be given, in order to adapt
    the model to the space in which the experiment defined environment operates. Usually, experiment name (to act as
    scope) and observation and action spaces types and sizes are required.

    Each rebuilt model (even belonging to the same instance) will have a different tensorflow scope in the graph
    depending on the experiment name.
    It's like the instance defines the type of model and its hyperparameters, while the generation of the model
    prepare such model for its effective use in a given experiment (which operates on an environment with certain
    observation and action spaces).

    Attributes:
        - name: name of the model
        - initializer: tensorflow operation to run to prepare the graph for training
        - summary: tensorflow summary to use in tensorboard
        - observation_space_shape: the shape of the observation space w.r.t. the environment on which the model is currently operating
        - action_space_shape: the shape of the action space w.r.t. the environment on which the model is currently operating
    """

    def __init__(self,
                 name: str):
        # Define attributes
        self.name: str = name
        # Initialize environment attributes (they require the model to be generated to be effective)
        # Those attributes will be reset each time the model is rebuilt for a new environment
        self._supported_observation_space_types: [] = []
        self._supported_action_space_types: [] = []
        self._experiment_name: str = None
        self._observation_space_type: SpaceType = None
        self.observation_space_shape = None
        self._action_space_type: SpaceType = None
        self.action_space_shape = None
        # Define empty general models attributes
        self.initializer = None
        self.summary = None

    def generate(self,
                 logger: logging.Logger,
                 scope: str,
                 observation_space_type: SpaceType, observation_space_shape,
                 action_space_type: SpaceType, action_space_shape) -> bool:
        """
        Generate the tensorflow model with the scope given by the format: experiment_name/model_name
        It calls the define (which is implemented on the child class) and define_summary methods.

        This method should be called each time the model is used in a different environment, since it makes sure
        that the model is rebuilt according to the environment scope, observations and actions spaces.

        :param scope: the str name of experiment/agent to use as a scope for the graph
        :param observation_space_type: the type of the environment observation space: discrete or continuous
        :param observation_space_shape: the shape of the environment observation space (it's a size if mono-dimensional)
        :param action_space_type: the type of the environment action space: discrete or continuous
        :param action_space_shape: the shape of the environment action space (it's a size if mono-dimensional)
        :param logger: the logger to use to record model generation information
        :return: True if the model generation is successful, False otherwise
        """
        self._observation_space_type = observation_space_type
        self.observation_space_shape = observation_space_shape
        self._action_space_type = action_space_type
        self.action_space_shape = action_space_shape
        self._experiment_name = scope
        logger.info("Generating model " + self.name + "...")
        # Check whether or not the observation space and the action space types are supported by the model
        observation_space_type_supported: bool = False
        for space_type in self._supported_observation_space_types:
            if self._observation_space_type == space_type:
                observation_space_type_supported = True
                break
        if not observation_space_type_supported:
            logger.error("Error during generation of model: observation space type not supported")
            return False
        action_space_type_supported: bool = False
        for space_type in self._supported_action_space_types:
            if self._action_space_type == space_type:
                action_space_type_supported = True
                break
        if not action_space_type_supported:
            logger.error("Error during generation of model: action space type not supported")
            return False
        logger.info("Model generation successful")
        # Define the tensorflow model graph
        self._define_graph()
        # Define the tensorflow summary for tensorboard
        self._define_summary()
        # Returns True to state success
        return True

    def _define_graph(self):
        """
        Define the tensorflow graph of the model.

        It uses as scope the format: experiment_name/agent_name/model_name
        """
        # Empty method, definition should be implemented on a child class basis
        pass

    def _define_summary(self):
        """
        Define the tensorboard summary of the model.

        It uses as scope the format: experiment_name/agent_name/model_name
        """
        # Empty method, definition should be implemented on a child class basis
        pass

    def initialize(self,
                   logger: logging.Logger,
                   session):
        """
        Initialize the variables of the model given the session.
        TODO: probably can add here from model already trained somehow

        :param logger: the logger to use to record model generation information
        :param session: the session of tensorflow currently running
        """
        # Initialize the model running the session on the appropriate tensorflow operation
        session.run(self.initializer)

    def predict(self,
                session,
                observation_current: numpy.ndarray):
        """
        Get the best action predicted by the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current state in the environment to get the prediction for
        :return: the best action predicted by the model
        """
        # Empty method, definition should be implemented on a child class basis
        pass

    def update(self,
               session,
               current_episode: int, total_episodes: int, current_step: int,
               batch: [], weights: numpy.ndarray):
        """
        Update the model weights (thus training the model) given a batch of samples, relative weights and a set of
        additional experiment parameters.

        :param session: the session of tensorflow currently running
        :param current_episode: the current episode number in the experiment
        :param total_episodes: the total episodes number in the experiment
        :param current_step: the current step number in the experiment
        :param batch: a batch of samples each one consisting of a tuple at least comprising: (state_current, action, reward, state_next)
        :param weights: the weights of each sample in the batch in the shape of a numpy array (ndarray)
        :return: the updated summary and a set of parameters (losses, errors, etc) depending on the model
        """
        # Empty method, definition should be implemented on a child class basis
        return None

    def get_trainable_variables(self,
                                scope: str):
        """
        Get the trainable variables in the model (useful for saving the model or comparing the weights), given the
        current experiment/agent scope.

        :param scope: the string scope of the tensorflow graph (usually experiment/agent with their respective names)
        :return: the trainable tensorflow variables.
        """
        # Get the training variables of the model under its scope: usually, the training variables of the tensorflow graph
        return tensorflow.trainable_variables(scope + "/" + self.name + "/")

    def print_trainable_variables(self,
                                  scope: str,
                                  session):
        """
        Print the trainable variables in the current tensorflow graph, given the current experiment/agent scope.

        :param scope: the string scope of the tensorflow graph (usually experiment/agent with their respective names)
        :param session: the session of tensorflow currently running
        """
        # Print the trainable variables of the currently active model
        trainable_variables = self.get_trainable_variables(scope)
        for trainable_variable in trainable_variables:
            print(trainable_variable.name)
            print(trainable_variable.eval(session=session))
