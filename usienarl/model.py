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

    Before use, a model needs to be generated. This define the tensorflow graph beneath the model.

    You should always define your own model, the base class cannot be used as-is.
    """

    def __init__(self,
                 name: str):
        # Define attributes
        self._name: str = name
        # Define empty attributes
        self._initializer = None
        self._scope: str or None = None
        self._parallel: int or None = None
        self._observation_space_type: SpaceType or None = None
        self._observation_space_shape: () = None
        self._agent_action_space_type: SpaceType or None = None
        self._agent_action_space_shape: () = None
        self._supported_observation_space_types: [] = []
        self._supported_action_space_types: [] = []

    def generate(self,
                 logger: logging.Logger,
                 scope: str,
                 parallel: int,
                 observation_space_type: SpaceType, observation_space_shape: (),
                 agent_action_space_type: SpaceType, agent_action_space_shape: ()) -> bool:
        """
        Generate the tensorflow model with the given scope.

        :param logger: the logger used to print the model information, warnings and errors
        :param scope: the scope of the graph to prepend to the model name
        :param parallel: the amount of parallel episodes run by the model
        :param observation_space_type: the type of the agent observation space
        :param observation_space_shape: the shape of the agent observation space wrapped in a tuple
        :param agent_action_space_type: the type of the agent action space
        :param agent_action_space_shape: the shape of the agent action space wrapped in a tuple
        :return True if the model generation is successful, False otherwise
        """
        # Make sure parameters are valid
        assert(scope is not None and scope)
        assert(parallel > 0)
        assert(observation_space_shape is not None and len(observation_space_shape) > 0)
        assert(agent_action_space_shape is not None and len(observation_space_shape) > 0)
        # Print info
        logger.info("Generating model " + self._name + " with scope " + scope + "...")
        # Set model attributes
        self._observation_space_type = observation_space_type
        self._observation_space_shape = observation_space_shape
        self._agent_action_space_type = agent_action_space_type
        self._agent_action_space_shape = agent_action_space_shape
        self._scope = scope
        self._parallel = parallel
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
        # Returns True to state success
        return True

    def initialize(self,
                   logger: logging.Logger,
                   session):
        """
        Initialize the tensorflow model.

        :param logger: the logger used to print the model information, warnings and errors
        :param session: the session of tensorflow currently running
        """
        # Initialize the model running the session on the appropriate tensorflow operation
        session.run(self._initializer)
        logger.info("Model initialized to default state")

    def _define_graph(self):
        """
        Define the tensorflow graph of the model.
        It uses as scope the format: experiment_name/agent_name/model_name
        """
        # Abstract method, definition should be implemented on a child class basis
        raise NotImplementedError()

    @property
    def name(self) -> str:
        """
        The name of the model.
        """
        return self._name

    @property
    def scope(self) -> str or None:
        """
        The scope of the model.
        It is None if model is not generated.
        """
        return self._scope

    @property
    def parallel(self) -> int or None:
        """
        The number of parallel episodes run by the model.
        It is None if model is not generated.
        """
        return self._parallel

    @property
    def observation_space_type(self) -> SpaceType or None:
        """
        The type of the observation space of the model. It is the same of the encompassing agent.
        It is None if model is not generated.
        """
        return self._observation_space_type

    @property
    def observation_space_shape(self) -> () or None:
        """
        The shape of the observation space of the model. It is the same of the encompassing agent.
        Note: it may differ from the environment's state space shape.
        It is None if model is not generated.
        """
        return self._observation_space_shape

    @property
    def action_space_type(self) -> SpaceType or None:
        """
        The type of the action space of the model. It is the same of the encompassing agent.
        It is None if model is not generated.
        """
        return self._agent_action_space_type

    @property
    def action_space_shape(self) -> () or None:
        """
        The shape of the action space of the model. It is the same of the encompassing agent.
        Note: it may differ from the environment's action space shape.
        It is None if model is not generated.
        """
        return self._agent_action_space_shape

    @property
    def trainable_variables(self):
        """
        The trainable variables in the mode, under 'scope/name'.
        """
        # Get the training variables of the model under its scope: usually, the training variables of the tensorflow graph
        return tensorflow.trainable_variables(self._scope + "/" + self._name + "/")

    @property
    def warmup_steps(self) -> int:
        """
        The number of steps required to warm-up the model.
        """
        # Abstract property, it should be implemented on a child class basis
        raise NotImplementedError()
