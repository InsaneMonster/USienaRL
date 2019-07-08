

class Memory:
    """
    Base class for each memory. It should not be used by itself, and should be extended instead.

    When a memory is created, it should be assigned to the experiment in order to be used by the _model,
    also assigned to the experiment.

    The batch size is defined in the experiment and it's not a property of the memory per-se.

    Attributes:
        - _capacity: integer representing the maximum size of the memory in units of samples
        - _warmup: boolean flag defining whether or not the memory requires to be pre-trained or not.
    """

    def __init__(self,
                 capacity: int, pre_train: bool):
        self.capacity: int = capacity
        self.pre_train: bool = pre_train

    def reset(self):
        """
        Reset the memory to its default empty state.
        """
        # Empty method, definition should be implemented on a child class basis
        pass

    def add_sample(self,
                   sample: []):
        """
        Add a sample to the memory. A sample usually consist of a list of current state, action, reward and next state.

        :param sample: the sample to add to the memory
        """
        # Empty method, definition should be implemented on a child class basis
        pass

    def get_sample(self,
                   amount: int = 0):
        """
        Get one or more samples from the memory, if any.

        :param amount: the amount of samples to get
        """
        # Empty method, definition should be implemented on a child class basis
        pass

    def update(self,
               absolute_error):
        """
        Update the memory using the absolute error from the _model.

        :param absolute_error: the absolute error from the _model
        """
        # Empty method, definition should be implemented on a child class basis
        pass
