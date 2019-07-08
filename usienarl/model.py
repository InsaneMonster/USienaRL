# Import packages

import tensorflow
import logging

# Import required src

from usienarl import SpaceType


class Model:
    """
    Base class for each _model. It should not be used by itself, and should be extended instead.

    When creating an instance of a _model (of a subclass of it) the _model is not yet defined.
    To define a proper tensorflow graph additional data regarding the experiment should be given, in order to adapt
    the _model to the space in which the experiment defined environment operates. Usually, experiment _name (to act as
    _scope) and observation and action spaces types and sizes are required.

    Each rebuilt _model (even belonging to the same instance) will have a different tensorflow _scope in the graph
    depending on the experiment _name.
    It's like the instance defines the type of _model and its hyperparameters, while the generation of the _model
    prepare such _model for its effective use in a given experiment (which operates on an environment with certain
    observation and action spaces).

    Attributes:
        - _name: name of the model
        - _initializer: tensorflow operation to run to prepare the graph for training
        - _summary: tensorflow summary to use in tensorboard
        - _observation_space_shape: the shape of the observation space w.r.t. the environment on which the model is currently operating
        - _action_space_shape: the shape of the action space w.r.t. the environment on which the model is currently operating
    """

    def __init__(self,
                 name: str):
        # Define _model attributes
        self._name: str = name
        # Define empty _model attributes
        self._initializer = None
        self._summary = None
        self._scope: str = None
        self._observation_space_type: SpaceType = None
        self._observation_space_shape = None
        self._action_space_type: SpaceType = None
        self._action_space_shape = None
        self._supported_observation_space_types: [] = []
        self._supported_action_space_types: [] = []

    def generate(self,
                 logger: logging.Logger,
                 scope: str,
                 observation_space_type: SpaceType, observation_space_shape,
                 action_space_type: SpaceType, action_space_shape) -> bool:
        """
        Generate the tensorflow model with the scope given by the format: experiment_name/agent_name/model_name
        It calls the define (which is implemented on the child class) and define _summary methods.

        This method is called every time the model is used in an agent on a new experiment or a new experiment iteration,
        since it makes sure that the model is rebuilt according to the experiment/agent scope, agent observations and
        actions spaces.

        :param logger: the logger used to print the _model information, warnings and errors
        :param scope: the str _name of experiment/agent to use as a _scope for the graph
        :param observation_space_type: the type of the agent observation space: discrete or continuous
        :param observation_space_shape: the shape of the agent observation space (it's a size if mono-dimensional)
        :param action_space_type: the type of the agent action space: discrete or continuous
        :param action_space_shape: the shape of the agent action space (it's a size if mono-dimensional)
        :return: True if the _model generation is successful, False otherwise
        """
        logger.info("Generating model " + self._name + " with scope " + scope + "...")
        # Set _model attributes
        self._observation_space_type = observation_space_type
        self._observation_space_shape = observation_space_shape
        self._action_space_type = action_space_type
        self._action_space_shape = action_space_shape
        self._scope = scope
        # Check whether or not the observation space and the action space types are supported by the _model
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
        # Define the tensorflow _model graph
        self._define_graph()
        # Define the tensorflow _summary for tensorboard
        self._define_summary()
        # Returns True to state success
        return True

    def initialize(self,
                   logger: logging.Logger,
                   session):
        """
        Initialize the variables of the model given the session.
        TODO: probably can add here from model already trained somehow

        :param logger: the logger used to print the _model information, warnings and errors
        :param session: the session of tensorflow currently running
        """
        # Initialize the _model running the session on the appropriate tensorflow operation
        session.run(self._initializer)
        logger.info("Model initialized to default state")

    @property
    def trainable_variables(self):
        """
        Get the trainable variables in the model (useful for saving the model or comparing the weights) given the
        current experiment/agent scope.

        :return: the trainable tensorflow variables.
        """
        # Get the training variables of the _model under its _scope: usually, the training variables of the tensorflow graph
        return tensorflow.trainable_variables(self._scope + "/" + self._name + "/")

    def _define_graph(self):
        """
        Define the tensorflow graph of the model.

        It uses as _scope the format: experiment_name/agent_name/model_name
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def _define_summary(self):
        """
        Define the tensorboard_summary of the model.

        It uses as _scope the format: experiment_name/agent_name/model_name
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def get_all_actions(self,
                        session,
                        observation_current):
        """
        Get all the actions values according to the _model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :return: all action values predicted by the model
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def get_best_action(self,
                        session,
                        observation_current):
        """
        Get the best action predicted by the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :return: the action predicted by the_model
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def get_best_action_and_all_actions(self,
                                        session,
                                        observation_current):
        """
        Get the best action predicted by the _model at the given current observation and all the action values according
        to the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :return: the best action predicted by the model and all action values predicted by the model
        """
        raise NotImplementedError()

    def update(self,
               session,
               batch: []):
        """
        Update the model weights (thus training the model) given a batch of samples, relative weights and a set of
        additional experiment parameters.

        :param session: the session of tensorflow currently running
        :param batch: a batch of samples each one consisting of a tuple at least comprising observation current, action and reward all wrapped in numpy arrays
        :return: the updated summary and a set of parameters (losses, errors, etc) depending on the model
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def warmup_episodes(self) -> int:
        """
        Get the number of episodes required to warm-up the model.

        :return: the integer number of warm-up episodes required by the model
        """
        raise NotImplementedError()
