# Import packages

import numpy

# Import required src

from usienarl import QLearningModel, Environment, Explorer


class BoltzmannExplorer(Explorer):
    """
    Boltzmann explorer using the model output to compute a probability distribution of the best state to visit.

    An output value from the predicted one by the model at the current state is used, and the randomness follows a
    distribution which is defined by the softmax of the same output values w.r.t. the current exploration rate value.
    """

    def __init__(self,
                 exploration_rate_start_value: float, exploration_rate_end_value: float,
                 exploration_rate_value_decay: float):
        # Generate the base explorer
        super().__init__(exploration_rate_start_value, exploration_rate_end_value, exploration_rate_value_decay)

    def get_action(self,
                   exploration_rate_current_value: float,
                   model: QLearningModel, environment: Environment, session, state_current: int) -> []:
        """
        Overridden method of Explorer class: check its docstring for further information.
        """
        # Choose an action according to the boltzmann approach
        # Get the model output and execute softmax on all the array components
        output = model.get_output(session, state_current)
        output = self._softmax(output / exploration_rate_current_value)
        # Get a random action value (random output) using the softmax as probability distribution
        action_value = numpy.random.choice(output[0], p=output[0])
        # Return the chose action as the index of such chosen action value
        return [numpy.argmax(output[0] == action_value)]

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
