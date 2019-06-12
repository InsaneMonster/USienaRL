# Import packages

import numpy

# Import required src

from usienarl import Memory, SumTree


class PrioritizedExperienceReplay(Memory):
    """
    Prioritized experience replay memory. It uses a sum tree to store not only the samples but also the priority of
    the samples, where the priority of the samples is a representation of the probability of a sample.

    When adding a new sample, the default priority value of the new node associated with such sample is set to
    the maximum defined priority. This value is iteratively changed by the algorithm when update is called.

    When getting the samples, a priority segment inversely proportional to the amount of required samples is generated.
    Samples are then uniformly taken for that segment, and stored in a minibatch which is returned.
    Also, a weight to compensate for the over-presence of higher priority samples is returned by the get_sample method,
    along with the minibatch as second returned value.
    """

    _MINIMUM_ALLOWED_PRIORITY: float = 1
    _IMPORTANCE_SAMPLING_VALUE_UPPER_BOUND: float = 1
    _ABSOLUTE_ERROR_UPPER_BOUND: float = 1

    def __init__(self,
                 capacity: int, pre_train: bool,
                 minimum_sample_probability: float, random_sample_trade_off: float,
                 importance_sampling_value: float, importance_sampling_value_increment: float):
        # Define prioritized experience replay attributes
        self._minimum_sample_probability: float = minimum_sample_probability
        self._random_sample_trade_off: float = random_sample_trade_off
        self._importance_sampling_value_increment: float = importance_sampling_value_increment
        self._importance_sampling_starting_value: float = importance_sampling_value
        self._importance_sampling_value = None
        self._sum_tree = None
        self._sum_tree_last_sampled_indexes = None
        # Generate the base memory
        super().__init__(capacity, pre_train)

    def reset(self):
        """
        Overridden method of Memory class: check its docstring for further information.
        """
        # Reset the last sampled indexes, the sum tree and the importance sampling value
        self._importance_sampling_value = self._importance_sampling_starting_value
        self._sum_tree_last_sampled_indexes = None
        self._sum_tree = SumTree(self.capacity)

    def add_sample(self,
                   sample: []):
        """
        Overridden method of Memory class: check its docstring for further information.
        """
        # Find the current max priority on the tree leafs
        max_priority: float = numpy.max(self._sum_tree.leafs)
        # If the max priority is zero set it to the minimum defined
        if max_priority <= 0:
            max_priority = self._MINIMUM_ALLOWED_PRIORITY
        # Set the max priority as the default one for this new sample
        # Note: we set the max priority for each new sample and then improve on it iteratively during training
        self._sum_tree.add(sample, max_priority)

    def get_sample(self,
                   amount: int = 0):
        """
        Overridden method of Memory class: check its docstring for further information.
        """
        # Create a sample array that will contains the minibatch
        minibatch: [] = []
        # Define the returned arrays of indexes and weights
        self._sum_tree_last_sampled_indexes = numpy.empty((amount, ), dtype=numpy.int32)
        minibatch_importance_sampling_weights = numpy.empty((amount, 1), dtype=numpy.float32)
        # Get the segment of total priority according to the given amount
        # Note: it divides the sum tree priority by the amount and get the priority assigned to each segment
        priority_segment: float = self._sum_tree.total_priority / amount
        # Increase the importance sampling value of the defined increment value until the upper bound is reached
        self._importance_sampling_value = numpy.min((self._IMPORTANCE_SAMPLING_VALUE_UPPER_BOUND,
                                                    self._importance_sampling_value + self._importance_sampling_value_increment))
        # Compute max importance sampling weight
        # Note: the weight of a given transition is inversely proportional to the probability of the transition stored
        # in the related leaf. The transition probability is computed by normalizing the priority of a leaf with the
        # total priority of the sum tree
        min_probability = numpy.min(self._sum_tree.leafs / self._sum_tree.total_priority)
        max_weight = (min_probability * amount) ** (-self._importance_sampling_value)
        # Return the sample
        for sample in range(amount):
            # Sample a random uniform value between the first and the last priority values of each priority segment
            lower_bound: float = priority_segment * sample
            upper_bound: float = priority_segment * (sample + 1)
            priority_value: float = numpy.random.uniform(lower_bound, upper_bound)
            # Get leaf index and related priority and data as stored in the sum tree
            leaf_index: int = self._sum_tree.get(priority_value)
            leaf_priority: float = self._sum_tree.get_priority(leaf_index)
            leaf_data = self._sum_tree.get_data(leaf_index)
            # Get the probability of the current sample
            sample_probability: float = leaf_priority / self._sum_tree.total_priority
            # Compute the importance sampling weights of each delta
            # The operation is: wj = (1/N * 1/P(j))**b / max wi == (N * P(j))**-b / max wi
            exponent: float = -self._importance_sampling_value
            minibatch_importance_sampling_weights[sample, 0] = ((sample_probability * amount)
                                                                ** exponent) / max_weight
            # Add the leaf index to the last sampled indexes list
            self._sum_tree_last_sampled_indexes[sample] = leaf_index
            minibatch.append(leaf_data)
        # Return the sample (minibatch) with related weights
        return minibatch, minibatch_importance_sampling_weights

    def update(self,
               absolute_errors: []):
        """
        Overridden method of Memory class: check its docstring for further information.
        """
        # If no last sampled indexes are found, stop here
        if self._sum_tree_last_sampled_indexes is None:
            return
        # Avoid absolute error (delta) equal to zero (which would result in zero priority), by adding an epsilon
        absolute_errors += self._minimum_sample_probability
        # Force an upper bound of 1 on each absolute error (delta + epsilon)
        absolute_errors = numpy.minimum(absolute_errors, self._ABSOLUTE_ERROR_UPPER_BOUND)
        # Compute the priority to store as (delta + epsilon)^alpha
        priority_values = absolute_errors ** self._random_sample_trade_off
        # Get only the max priority values along each row (second axis)
        # Note: this is necessary since the absolute error is always zero between the current outputs and target outputs
        # when the action index is not the same of the chosen action of the sample
        priority_values = numpy.amax(priority_values, axis=1)
        # Before zipping reshape the absolute error array to be compatible with the stored tree indexes
        priority_values.reshape(self._sum_tree_last_sampled_indexes.shape)
        # For each last sampled sum tree leaf index and each correlated priority value update the sum tree
        for sum_tree_index, priority_value in zip(self._sum_tree_last_sampled_indexes, priority_values):
            self._sum_tree.update(sum_tree_index, priority_value)
        # Reset the last sampled sum tree indexes for the next update
        self._sum_tree_last_sampled_indexes = None
