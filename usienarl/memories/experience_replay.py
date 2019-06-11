# Import packages

import random
import numpy

# Import required src

from usienarl import Memory


class ExperienceReplay(Memory):
    """
    Experience replay memory. It consist of a simple list of samples from which to get samples at random.

    If too many samples are inserted, the first one inserted is removed (like a FIFO system).
    If more samples are required than there are stored in memory, only as many as the current size of the memory
    are returned.

    Does not require to be updated.
    """

    def __init__(self,
                 capacity: int, pre_train: bool):
        # Define experience replay attributes
        self._samples = None
        # Generate the base memory
        super().__init__(capacity, pre_train)

    def reset(self):
        """
        Overridden method of Memory class: check its docstring for further information.
        """
        # Reset the samples
        self._samples = []

    def add_sample(self,
                   sample: []):
        """
        Overridden method of Memory class: check its docstring for further information.
        """
        # Add a new sample and, if capacity is exceeded, remove the oldest (first inserted) sample
        self._samples.append(sample)
        if len(self._samples) > self.capacity:
            self._samples.pop(0)

    def get_sample(self,
                   amount: int = 0):
        """
        Overridden method of Memory class: check its docstring for further information.
        """
        # Select random samples up to the length of the samples array
        # Return the sample with associated weights (all weights are equal and just one)
        if amount > len(self._samples):
            return random.sample(self._samples, len(self._samples)), numpy.ones((len(self._samples), 1))
        else:
            return random.sample(self._samples, amount), numpy.ones((amount, 1))
