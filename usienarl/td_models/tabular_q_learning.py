# Import packages

import tensorflow
import numpy

# Import required src

from usienarl import SpaceType, Model
from usienarl.utils import SumTree


class Buffer:
    """
    Prioritized experience replay buffer. It uses a sum tree to store not only the samples but also the priority of
    the samples, where the priority of the samples is a representation of the probability of a sample.

    When storing a new sample, the default priority value of the new node associated with such sample is set to
    the maximum defined priority. This value is iteratively changed by the algorithm when update is called.

    When getting the samples, a priority segment inversely proportional to the amount of required samples is generated.
    Samples are then uniformly taken for that segment, and stored in a minibatch which is returned.
    Also, a weight to compensate for the over-presence of higher priority samples is returned by the get method,
    along with the minibatch as second returned value.
    """

    _MINIMUM_ALLOWED_PRIORITY: float = 1.0
    _IMPORTANCE_SAMPLING_VALUE_UPPER_BOUND: float = 1.0
    _ABSOLUTE_ERROR_UPPER_BOUND: float = 1.0

    def __init__(self,
                 capacity: int,
                 minimum_sample_probability: float, random_sample_trade_off: float,
                 importance_sampling_value: float, importance_sampling_value_increment: float):
        # Define internal prioritized experience replay buffer attributes
        self._capacity: int = capacity
        self._minimum_sample_probability: float = minimum_sample_probability
        self._random_sample_trade_off: float = random_sample_trade_off
        self._importance_sampling_value_increment: float = importance_sampling_value_increment
        self._importance_sampling_starting_value: float = importance_sampling_value
        self._sum_tree = SumTree(self._capacity)
        self._importance_sampling_value = self._importance_sampling_starting_value
        # Define internal prioritized experience replay buffer empty attributes
        self._sum_tree_last_sampled_indexes = None

    def store(self,
              observation_current, action, reward: float, observation_next, last_step: bool):
        # Find the current max priority on the tree leafs
        max_priority: float = numpy.max(self._sum_tree.leafs)
        # If the max priority is zero set it to the minimum defined
        if max_priority <= 0:
            max_priority = self._MINIMUM_ALLOWED_PRIORITY
        # Set the max priority as the default one for this new sample
        # Note: we set the max priority for each new sample and then improve on it iteratively during training
        self._sum_tree.add((observation_current, action, reward, observation_next, last_step), max_priority)

    def get(self,
            amount: int = 0):
        # Adjust amount with respect to the size of the sum-tree
        if amount <= 0 or amount > self._sum_tree.size:
            amount = self._sum_tree.size
        # Define arrays of each sample components
        observations_current: [] = []
        actions: [] = []
        rewards: [] = []
        observations_next: [] = []
        last_steps: [] = []
        # Define the returned arrays of indexes and weights
        self._sum_tree_last_sampled_indexes = numpy.empty((amount,), dtype=numpy.int32)
        importance_sampling_weights = numpy.empty((amount, 1), dtype=numpy.float32)
        # Get the segment of total priority according to the given amount
        # Note: it divides the sum tree priority by the amount and get the priority assigned to each segment
        priority_segment: float = self._sum_tree.total_priority / amount
        # Increase the importance sampling value of the defined increment value until the upper bound is reached
        self._importance_sampling_value = numpy.min((self._IMPORTANCE_SAMPLING_VALUE_UPPER_BOUND, self._importance_sampling_value + self._importance_sampling_value_increment))
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
            importance_sampling_weights[sample, 0] = ((sample_probability * amount) ** exponent) / max_weight
            # Add the leaf index to the last sampled indexes list
            self._sum_tree_last_sampled_indexes[sample] = leaf_index
            # Generate the minibatch for this example
            observations_current.append(leaf_data[0])
            actions.append(leaf_data[1])
            rewards.append(leaf_data[2])
            observations_next.append(leaf_data[3])
            last_steps.append(leaf_data[4])
        # Return the sample (minibatch) with related weights
        return [numpy.array(observations_current), numpy.array(actions), numpy.array(rewards), numpy.array(observations_next), numpy.array(last_steps), importance_sampling_weights]

    def update(self,
               absolute_errors: []):
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


# Define update rules for the model


class TabularQLearning(Model):
    """
    Tabular temporal difference model with Q-Learning update rule (Q-Table).
    The weights of the model are the entries of the table and the outputs are computed by multiplication of this
    matrix elements by the inputs.

    The update rule is the following (Bellman equation):
    Q(s, a) = R + gamma * max_a(Q(s'))
    It uses the max of the predicted q-value at the next state to update predicted q-value at current state

    Supported observation spaces:
        - discrete

    Supported action spaces:
        - discrete
    """

    def __init__(self,
                 name: str,
                 learning_rate: float, discount_factor: float,
                 buffer_capacity: int,
                 minimum_sample_probability: float, random_sample_trade_off: float,
                 importance_sampling_value: float, importance_sampling_value_increment: float):
        # Define model attributes
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        # Define internal model attributes
        self._buffer_capacity: int = buffer_capacity
        self._minimum_sample_probability: float = minimum_sample_probability
        self._random_sample_trade_off: float = random_sample_trade_off
        self._importance_sampling_value: float = importance_sampling_value
        self._importance_sampling_value_increment: float = importance_sampling_value_increment
        # Define model empty attributes
        self.buffer: Buffer = None
        # Define internal model empty attributes
        self._inputs = None
        self._mask = None
        self._outputs = None
        self._table = None
        self._targets = None
        self._loss_weights = None
        self._absolute_error = None
        self._loss = None
        self._optimizer = None
        # Generate the base model
        super(TabularQLearning, self).__init__(name)
        # Define the types of allowed observation and action spaces
        self._supported_observation_space_types.append(SpaceType.discrete)
        self._supported_action_space_types.append(SpaceType.discrete)

    def _define_graph(self):
        # Set the buffer
        self.buffer = Buffer(self._buffer_capacity, self._minimum_sample_probability, self._random_sample_trade_off,
                             self._importance_sampling_value, self._importance_sampling_value_increment)
        # Define the tensorflow model
        with tensorflow.variable_scope(self._scope + "/" + self._name):
            # Define inputs of the _model as a float adaptable array with size Nx(S) where N is the number of examples and (O) is the shape of the observations space
            self._inputs = tensorflow.placeholder(shape=[None, *self._observation_space_shape], dtype=tensorflow.float32, name="inputs")
            # Define mask placeholder
            self._mask = tensorflow.placeholder(shape=[None, *self._agent_action_space_shape], dtype=tensorflow.float32, name="mask")
            # Initialize the table (a weight matrix) of OxA dimensions with random uniform numbers between 0 and 0.1
            self._table = tensorflow.get_variable(name="table", trainable=True, initializer=tensorflow.random_uniform([*self._observation_space_shape, *self._agent_action_space_shape], 0, 0.1))
            # Define the outputs at a given state as a matrix of size NxA given by multiplication of inputs and weights
            # Define the targets for learning with the same Nx(A) adaptable size
            # Note: N is the number of examples and (A) the shape of the action space
            self._outputs = tensorflow.add(tensorflow.matmul(self._inputs, self._table), self._mask, name="outputs")
            self._targets = tensorflow.placeholder(shape=[None, *self._agent_action_space_shape], dtype=tensorflow.float32, name="targets")
            # Define the weights of the targets during the update process (e.g. the importance sampling weights)
            self._loss_weights = tensorflow.placeholder(shape=[None, 1], dtype=tensorflow.float32, name="loss_weights")
            # Define the absolute error
            self._absolute_error = tensorflow.abs(self._targets - self._outputs, name="absolute_error")
            # Define the loss
            self._loss = tensorflow.reduce_sum(self._loss_weights * tensorflow.squared_difference(self._targets, self._outputs), name="loss")
            # Define the optimizer
            self._optimizer = tensorflow.train.GradientDescentOptimizer(self.learning_rate).minimize(self._loss)
            # Define the initializer
            self._initializer = tensorflow.global_variables_initializer()

    def _define_summary(self):
        with tensorflow.variable_scope(self._scope + "/" + self._name):
            # Define the _summary operation for this graph with loss and absolute error summaries
            self._summary = tensorflow.summary.merge([tensorflow.summary.scalar("loss", self._loss)])

    def get_all_action_values(self,
                              session,
                              observation_current,
                              mask: numpy.ndarray = None):
        """
        Get all the actions values according to the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :param mask: the optional mask used to remove certain actions from the prediction (-infinity to remove, 0.0 to pass-through)
        :return: all action values predicted by the model
        """
        # If there is no mask generate a full pass-through mask
        if mask is None:
            mask = numpy.zeros(self._agent_action_space_shape, dtype=float)
        # Generate a one-hot encoded version of the observation
        observation_current_one_hot: numpy.ndarray = numpy.identity(*self._observation_space_shape)[observation_current]
        # Return all the predicted q-values given the current observation
        return session.run(self._outputs, feed_dict={self._inputs: [observation_current_one_hot], self._mask: [mask]})

    def get_best_action(self,
                        session,
                        observation_current,
                        mask: numpy.ndarray = None):
        """
        Get the best action predicted by the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :param mask: the optional mask used to remove certain actions from the prediction (-infinity to remove, 0.0 to pass-through)
        :return: the action predicted by the model
        """
        # Return the predicted action given the current observation
        return numpy.argmax(self.get_all_action_values(session, observation_current, mask))

    def get_best_action_and_all_action_values(self,
                                              session,
                                              observation_current,
                                              mask: numpy.ndarray = None):
        """
        Get the best action predicted by the model at the given current observation and all the action values according
        to the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :param mask: the optional mask used to remove certain actions from the prediction (-infinity to remove, 0.0 to pass-through)
        :return: the best action predicted by the model and all action values predicted by the model
        """
        # Get all actions
        all_actions = self.get_all_action_values(session, observation_current, mask)
        # Return the best action and all the actions
        return numpy.argmax(all_actions), all_actions

    def update(self,
               session,
               batch: []):
        # Generate a full pass-through mask for each example in the batch
        masks: numpy.ndarray = numpy.zeros((len(batch[0]), *self._agent_action_space_shape), dtype=float)
        # Unpack the batch into numpy arrays
        observations_current, actions, rewards, observations_next, last_steps, weights = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        # Generate a one-hot encoded version of the observations
        observations_current_one_hot: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observations_current.reshape(-1)]
        observations_next_one_hot: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observations_next.reshape(-1)]
        # Get the q-values from the model at both current observations and next observations
        q_values_current: numpy.ndarray = session.run(self._outputs, feed_dict={self._inputs: observations_current_one_hot, self._mask: masks})
        q_values_next: numpy.ndarray = session.run(self._outputs, feed_dict={self._inputs: observations_next_one_hot, self._mask: masks})
        # Apply Bellman equation with the Q-Learning update rule
        for sample_index in range(len(actions)):
            # Extract current sample values
            action = actions[sample_index]
            reward: float = rewards[sample_index]
            last_step: bool = last_steps[sample_index]
            # Note: only the immediate reward can be assigned at end of the episode, i.e. when next observation is None
            if last_step:
                q_values_current[sample_index, action] = reward
            else:
                q_values_current[sample_index, action] = reward + self.discount_factor * numpy.max(q_values_next[sample_index])
        # Train the model and save the value of the loss and of the absolute error as well as the summary
        _, loss, absolute_error, summary = session.run([self._optimizer, self._loss, self._absolute_error, self._summary],
                                                       feed_dict={
                                                                   self._inputs: observations_current_one_hot,
                                                                   self._targets: q_values_current,
                                                                   self._loss_weights: weights,
                                                                   self._mask: masks
                                                                  })
        # Return the loss, the absolute error and relative summary
        return summary, loss, absolute_error

    @property
    def warmup_steps(self) -> int:
        return self._buffer_capacity

    @property
    def inputs_name(self) -> str:
        # Get the _name of the inputs of the tensorflow graph
        return "inputs"

    @property
    def outputs_name(self) -> str:
        # Get the _name of the outputs of the tensorflow graph
        return "outputs"
