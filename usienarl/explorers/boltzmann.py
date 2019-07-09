# Import packages

import logging
import numpy

# Import required src

from usienarl import ExplorationPolicy, Interface, SpaceType


class BoltzmannExplorer(ExplorationPolicy):
    """
    TODO: summary
    """

    def __init__(self,
                 temperature_max: float, temperature_min: float,
                 temperature_decay: float):
        # Define boltzmann exploration policy attributes
        self._temperature_max: float = temperature_max
        self._temperature_min: float = temperature_min
        self._temperature_decay: float = temperature_decay
        # Define epsilon-greedy empty exploration policy attributes
        self._temperature: float = None
        # Generate the base explorer
        super().__init__()
        # Define the types of allowed action space types
        self._supported_action_space_types.append(SpaceType.discrete)

    def _define(self):
        pass

    def initialize(self, logger: logging.Logger, session):
        # Reset temperature to its starting value (the max)
        self._temperature = self._temperature_max

    def act(self,
            logger: logging.Logger,
            session,
            interface: Interface,
            all_actions, best_action):
        # Act according to boltzmann approach: get the softmax over all the actions predicted by the model
        output = self._softmax(all_actions / self._temperature)
        # Get a random action value (random output) using the softmax as probability distribution
        action_value = numpy.random.choice(output[0], p=output[0])
        # Return the chose action as the index of such chosen action value
        return numpy.argmax(output[0] == action_value)

    def update(self,
               logger: logging.Logger,
               session):
        # Decrease the exploration rate by its decay value
        self._temperature = max(self._temperature_min, self._temperature - self._temperature_decay)

    @staticmethod
    def _softmax(array):
        """
        Compute the softmax of an array.

        :param array: the array of which to compute the softmax
        :return: the softmax value
        """
        # Make sure the length of the shape of the given array is 2
        assert len(array.shape) == 2
        # Get the element with max value in the given array as the normalization factor
        normalization_factor = numpy.max(array, axis=1)
        # Increase the shape size of the normalization factor to allow broadcasting
        normalization_factor = normalization_factor[:, numpy.newaxis]
        # Apply the normalization
        numerator = numpy.exp(array - normalization_factor)
        # Compute the denominator by summing all the normalized numerator elements
        denominator = numpy.sum(numerator, axis=1)
        # Increase the shape size of the denominator to allow broadcasting
        denominator = denominator[:, numpy.newaxis]
        # Return the softmax result
        return numerator / denominator
