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

import tensorflow
import logging

# Import required src

from usienarl import SpaceType


class Model:
    """
    Base abstract model class.
    A model is whatever is moving an agent, deciding actions according to one or more policies and being updated to
    better learn those policies.

    Before use, a model needs to be generated. This will make sure to define the internal structure of the model
    (usually a tensorflow graph).

    To define your own model, implement the abstract class in a specific child class.
    """

    def __init__(self,
                 name: str):
        # Define model attributes
        self._name: str = name
        # Define empty model attributes
        self._initializer = None
        self._summary = None
        self._scope: str = None
        self._observation_space_type: SpaceType = None
        self._observation_space_shape = None
        self._agent_action_space_type: SpaceType = None
        self._agent_action_space_shape = None
        self._supported_observation_space_types: [] = []
        self._supported_action_space_types: [] = []

    def generate(self,
                 logger: logging.Logger,
                 scope: str,
                 observation_space_type: SpaceType, observation_space_shape,
                 agent_action_space_type: SpaceType, agent_action_space_shape) -> bool:
        """
        Generate the tensorflow model with the scope given by the format: experiment_name/agent_name/model_name
        It calls the define (which is implemented on the child class) and define summary methods.

        This method is called every time the model is used in an agent on a new experiment or a new experiment iteration,
        since it makes sure that the model is rebuilt according to the experiment/agent scope, agent observations and
        actions spaces.

        :param logger: the logger used to print the model information, warnings and errors
        :param scope: the str _name of experiment/agent to use as a scope for the graph
        :param observation_space_type: the type of the agent observation space: discrete or continuous
        :param observation_space_shape: the shape of the agent observation space (it's a size if mono-dimensional)
        :param agent_action_space_type: the type of the agent action space: discrete or continuous
        :param agent_action_space_shape: the shape of the agent action space (it's a size if mono-dimensional)
        :return: True if the model generation is successful, False otherwise
        """
        logger.info("Generating model " + self._name + " with scope " + scope + "...")
        # Set model attributes
        self._observation_space_type = observation_space_type
        self._observation_space_shape = observation_space_shape
        self._agent_action_space_type = agent_action_space_type
        self._agent_action_space_shape = agent_action_space_shape
        self._scope = scope
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
            if self._agent_action_space_type == space_type:
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

    def initialize(self,
                   logger: logging.Logger,
                   session):
        """
        Initialize the variables of the model given the session.

        :param logger: the logger used to print the model information, warnings and errors
        :param session: the session of tensorflow currently running
        """
        # Initialize the model running the session on the appropriate tensorflow operation
        session.run(self._initializer)
        logger.info("Model initialized to default state")

    @property
    def trainable_variables(self):
        """
        Get the trainable variables in the model (useful for saving the model or comparing the weights) given the
        current experiment/agent scope.

        :return: the trainable tensorflow variables.
        """
        # Get the training variables of the model under its scope: usually, the training variables of the tensorflow graph
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
        Define the tensorboard summary of the model.

        It uses as scope the format: experiment_name/agent_name/model_name
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    def predict(self,
                session,
                observation_current):
        """
        Get the action predicted by the model given the current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :return: the action predicted by the model
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
