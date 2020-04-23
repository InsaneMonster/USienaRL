# Import packages

import tensorflow
import numpy
import math

# Import required src

from usienarl import SpaceType, Model, Config
from usienarl.utils import SumTree


class Buffer:
    """
    Prioritized experience replay buffer. It uses a sum tree to store not only the samples but also the priority of
    the samples, where the priority of the samples is a representation of the probability of a sample.

    When storing a new sample, the default priority value of the new node associated with such sample is set to
    the maximum defined priority. This value is iteratively changed by the algorithm when update is called.
    It automatically serializes all the parallel data stored in it.

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
                 parallel_amount: int,
                 minimum_sample_probability: float, random_sample_trade_off: float,
                 importance_sampling_value: float, importance_sampling_value_increment: float):
        # Define internal prioritized experience replay buffer attributes
        self._capacity: int = capacity
        self._parallel_amount: int = parallel_amount
        self._minimum_sample_probability: float = minimum_sample_probability
        self._random_sample_trade_off: float = random_sample_trade_off
        self._importance_sampling_value_increment: float = importance_sampling_value_increment
        self._importance_sampling_starting_value: float = importance_sampling_value
        self._sum_tree = SumTree(self._capacity)
        self._importance_sampling_value = self._importance_sampling_starting_value
        # Define internal prioritized experience replay buffer empty attributes
        self._sum_tree_last_sampled_indexes = None
        self._episode_done_previous_step: numpy.ndarray = numpy.zeros(self._parallel_amount, dtype=bool)

    def store(self,
              observation_current: numpy.ndarray,
              action_current: numpy.ndarray,
              reward: numpy.ndarray,
              observation_next: numpy.ndarray,
              action_next: numpy.ndarray,
              episode_done: numpy.ndarray):
        """
        Store the time-step in the buffer, serializing immediately all the parallel episodes.

        :param observation_current: the current observation to store in the buffer wrapped in a numpy array
        :param action_current: the last current action to store in the buffer wrapped in a numpy array
        :param reward: the reward obtained from the action at the current state to store in the buffer wrapped in a numpy array
        :param observation_next: the next observation to store in the buffer wrapped in a numpy array
        :param action_next: the last next action to store in the buffer wrapped in a numpy array
        :param episode_done: whether or not this time-step was the last of the episode wrapped in a numpy array
        """
        # Serialize all the experiences to store in the buffer
        for i in range(self._parallel_amount):
            if self._episode_done_previous_step[i]:
                continue
            # Find the current max priority on the tree leafs
            max_priority: float = numpy.max(self._sum_tree.leafs)
            # If the max priority is zero set it to the minimum defined
            if max_priority <= 0:
                max_priority = self._MINIMUM_ALLOWED_PRIORITY
            # Set the max priority as the default one for this new sample
            # Note: we set the max priority for each new sample and then improve on it iteratively during training
            self._sum_tree.add((observation_current[i], action_current[i], reward[i], observation_next[i], action_next[i], episode_done[i]), max_priority)
        # Update the stored episode done flags
        self._episode_done_previous_step = episode_done.copy()

    def get(self,
            amount: int = 0):
        """
        Get a batch of data from the buffer of the given size. If size is not given all the buffer is used.

        :param amount: the batch size of data to get
        :return a list containing the arrays of: current observations, actions, rewards, next observations and episode done flags
        """
        # Adjust amount with respect to the size of the sum-tree
        if amount <= 0 or amount > self._sum_tree.size:
            amount = self._sum_tree.size
        # Define arrays of each sample components
        observations_current: [] = []
        actions_current: [] = []
        rewards: [] = []
        observations_next: [] = []
        actions_next: [] = []
        episode_done_flags: [] = []
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
            actions_current.append(leaf_data[1])
            rewards.append(leaf_data[2])
            observations_next.append(leaf_data[3])
            actions_next.append(leaf_data[4])
            episode_done_flags.append(leaf_data[5])
        # Return the sample (minibatch) with related weights
        return [numpy.array(observations_current), numpy.array(actions_current), numpy.array(rewards), numpy.array(observations_next), numpy.array(actions_next), numpy.array(episode_done_flags), importance_sampling_weights]

    def update(self,
               absolute_errors: []):
        """
        Update the buffer using the given absolute errors.

        :param absolute_errors: the absolute errors on the values predictions
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

    def finish_trajectory(self):
        """
        Finish the trajectory, resetting episode done flags.
        """
        #  Reset stored episode done flags
        self._episode_done_previous_step: numpy.ndarray = numpy.zeros(self._parallel_amount, dtype=bool)

    @property
    def capacity(self) -> int:
        """
        The capacity of the buffer..

        :return: the integer capacity of the buffer
        """
        return self._capacity

    @property
    def size(self) -> int:
        """
        The size of the buffer at the current time.

        :return: the integer size of the buffer
        """
        return self._sum_tree.size


class Estimator:
    """
    Estimator defining the real Deep SARSA model. It is used to define two identical models:
    target network and main-network.

    It is generated given the size of the observation and action spaces and the hidden layer config defining the
    hidden layers of the network.
    """

    def __init__(self,
                 scope: str,
                 observation_space_shape, agent_action_space_shape,
                 hidden_layers_config: Config,
                 error_clipping: bool = False,
                 huber_delta: float = 1.0):
        self.scope: str = scope
        with tensorflow.variable_scope(self.scope):
            # Define observations placeholder as a float adaptable array with shape Nx(O) where N is the number of examples and (O) the shape of the observation space
            self.observations = tensorflow.placeholder(shape=[None, *observation_space_shape], dtype=tensorflow.float32, name="observations")
            # Define the q-values targets placeholder with adaptable size NxA where N is the number of examples and A the size of the action space (always discrete)
            self.q_values_targets = tensorflow.placeholder(shape=[None, *agent_action_space_shape], dtype=tensorflow.float32, name="q_values_targets")
            # Define the estimator network hidden layers from the config
            hidden_layers_output = hidden_layers_config.apply_hidden_layers(self.observations)
            # Define the mask
            self.mask = tensorflow.placeholder(shape=[None, *agent_action_space_shape], dtype=tensorflow.float32, name="mask")
            # Define the estimator network predicted q-values given the input observations and the mask
            self.q_values_predictions = tensorflow.add(tensorflow.layers.dense(hidden_layers_output, *agent_action_space_shape, name="q_values_predictions"), self.mask)
            # Define the weights of the targets during the update process (e.g. the importance sampling weights)
            self.loss_weights = tensorflow.placeholder(shape=[None, 1], dtype=tensorflow.float32, name="loss_weights")
            # Define the absolute error
            self.absolute_error = tensorflow.abs(self.q_values_targets - self.q_values_predictions, name="absolute_error")
            # Define the loss with error clipping (huber loss) if required
            if error_clipping:
                self.loss = tensorflow.reduce_mean(self.loss_weights * tensorflow.where(self.absolute_error < huber_delta,
                                                   0.5 * tensorflow.square(self.absolute_error),
                                                   self.absolute_error - 0.5), name="loss")
            else:
                self.loss = tensorflow.reduce_mean(self.loss_weights * tensorflow.square(self.absolute_error), name="loss")
            # Define the estimator weight parameters
            self.weight_parameters = [variable for variable in tensorflow.trainable_variables() if variable.name.startswith(self.scope)]
            self.weight_parameters = sorted(self.weight_parameters, key=lambda parameter: parameter.name)


class DeepSARSA(Model):
    """
    Deep SARSA model with SARSA update rule.
    The model is a deep neural network whose hidden layers can be defined by a config parameter.
    It uses a target network and a main network to correctly evaluate the expected future reward in order to stabilize
    learning. To further stabilize learning error can be clipped through the Huber Loss (note that this may cause
    instability in some environments).

    In order to synchronize the target network and the main network, every some interval steps model weights have to be
    copied from the main network to the target network.

    The update rule is the following (Bellman equation):
    Q(s, a) = R + gamma * Q(s', a')
    It uses also the action predicted at the next observation according to the same policy to update the current
    estimate of q-values.

    Supported observation spaces:
        - discrete
        - continuous

    Supported action spaces:
        - discrete
    """

    def __init__(self,
                 name: str,
                 hidden_layers_config: Config,
                 buffer_capacity: int = 1000000,
                 learning_rate: float = 1e-3, discount_factor: float = 0.99,
                 minimum_sample_probability: float = 1e-2, random_sample_trade_off: float = 0.6,
                 importance_sampling_value: float = 1e-3, importance_sampling_value_increment: float = 0.4,
                 error_clipping: bool = False, huber_delta: float = 1.0):
        # Define model attributes
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        # Define internal model attributes
        self._buffer_capacity: int = buffer_capacity
        self._minimum_sample_probability: float = minimum_sample_probability
        self._random_sample_trade_off: float = random_sample_trade_off
        self._importance_sampling_value: float = importance_sampling_value
        self._importance_sampling_value_increment: float = importance_sampling_value_increment
        self._hidden_layers_config: Config = hidden_layers_config
        self._error_clipping: bool = error_clipping
        self._huber_delta: float = huber_delta
        # Define model empty attributes
        self.buffer: Buffer or None = None
        # Define internal model empty attributes
        self._target_network: Estimator or None = None
        self._main_network: Estimator or None = None
        self._target_network_observations = None
        self._target_network_q_values_predictions = None
        self._main_network_observations = None
        self._main_network_q_values_predictions = None
        self._main_network_mask = None
        self._target_network_mask = None
        self._q_values_targets = None
        self._loss_weights = None
        self._absolute_error = None
        self._loss = None
        self._optimizer = None
        self._weight_copier = None
        # Generate the base model
        super(DeepSARSA, self).__init__(name)
        # Define the types of allowed observation and action spaces
        self._supported_observation_space_types.append(SpaceType.discrete)
        self._supported_observation_space_types.append(SpaceType.continuous)
        self._supported_action_space_types.append(SpaceType.discrete)

    def _define_graph(self):
        # Set the buffer
        self.buffer = Buffer(self._buffer_capacity, self._parallel,
                             self._minimum_sample_probability, self._random_sample_trade_off,
                             self._importance_sampling_value, self._importance_sampling_value_increment)
        # Define two estimator, one for target network and one for main network, with identical structure
        full_scope: str = self._scope + "/" + self._name
        self._main_network = Estimator(full_scope + "/MainNetwork",
                                       self._observation_space_shape, self._agent_action_space_shape,
                                       self._hidden_layers_config,
                                       self._error_clipping, self._huber_delta)
        self._target_network = Estimator(full_scope + "/TargetNetwork",
                                         self._observation_space_shape, self._agent_action_space_shape,
                                         self._hidden_layers_config,
                                         self._error_clipping, self._huber_delta)
        # Assign main and target networks to the model attributes
        self._main_network_observations = self._main_network.observations
        self._main_network_q_values_predictions = self._main_network.q_values_predictions
        self._target_network_q_values_predictions = self._target_network.q_values_predictions
        self._target_network_observations = self._target_network.observations
        self._main_network_mask = self._main_network.mask
        self._target_network_mask = self._target_network.mask
        self._q_values_targets = self._main_network.q_values_targets
        self._absolute_error = self._main_network.absolute_error
        self._loss = self._main_network.loss
        self._loss_weights = self._main_network.loss_weights
        # Define global operations
        with tensorflow.variable_scope(full_scope):
            # Define the optimizer
            self._optimizer = tensorflow.train.AdamOptimizer(self.learning_rate).minimize(self._loss)
            # Define the initializer
            self._initializer = tensorflow.variables_initializer(tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES, full_scope), name="initializer")
            # Define the weight copier (to copy weights from main network to target network)
            self._weight_copier = []
            for main_network_parameter, target_network_parameter in zip(self._main_network.weight_parameters, self._target_network.weight_parameters):
                copy_operation = target_network_parameter.assign(main_network_parameter)
                self._weight_copier.append(copy_operation)

    def get_q_values(self,
                     session,
                     observation_current: numpy.ndarray,
                     possible_actions: numpy.ndarray = None):
        """
        Get all the q-values according to the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon, wrapped in a numpy array
        :param possible_actions: the optional array used to remove certain actions from the prediction
        :return: all q-values predicted by the model
        """
        # If there is no possible actions list generate a full pass-through mask otherwise generate a mask upon it
        if possible_actions is None:
            mask: numpy.ndarray = numpy.zeros((self._parallel, *self._agent_action_space_shape), dtype=float)
        else:
            mask: numpy.ndarray = -math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
            mask[:, possible_actions] = 0.0
        # Generate a one-hot encoded version of the observation if observation space is discrete
        if self._observation_space_type == SpaceType.discrete:
            observation_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[numpy.array(observation_current).reshape(-1)]
        # Compute the q-values predicted by main and target networks
        main_network_q_values = session.run(self._main_network_q_values_predictions,
                                            feed_dict={
                                                self._main_network_observations: observation_current,
                                                self._main_network_mask: mask
                                            })
        target_network_q_values = session.run(self._target_network_q_values_predictions,
                                              feed_dict={
                                                  self._target_network_observations: observation_current,
                                                  self._target_network_mask: mask
                                              })
        # Return the average of the q-values
        return (main_network_q_values + target_network_q_values) / 2

    def get_action_with_highest_q_value(self,
                                        session,
                                        observation_current: numpy.ndarray,
                                        possible_actions: numpy.ndarray = None):
        """
        Get the action with highest q-value from the q-values predicted by the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon, wrapped in a numpy array
        :param possible_actions: the optional array used to remove certain actions from the prediction
        :return: the action chosen by the model
        """
        # Return the action maximizing the predicted q-values given the current observation
        return numpy.argmax(self.get_q_values(session, observation_current, possible_actions), axis=1)

    def get_action_with_highest_q_value_and_q_values(self,
                                                     session,
                                                     observation_current: numpy.ndarray,
                                                     possible_actions: numpy.ndarray = None):
        """
        Get the action with highest q-value from the q-values predicted by the model at the given current observation
        and all the q-values according to the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon, wrapped in a numpy array
        :param possible_actions: the optional array used to remove certain actions from the prediction
        :return: the action chosen by the model and all q-values predicted by the model
        """
        # Get q-values
        q_values = self.get_q_values(session, observation_current, possible_actions)
        # Return the highest q-value action and all the q-values
        return numpy.argmax(q_values, axis=1), q_values

    def copy_weights(self,
                     session):
        """
        Copy the weights from the main network to the target network.

        :param session: the session of tensorflow currently running
        """
        # Run all the weight copy operations
        session.run(self._weight_copier)

    def update(self,
               session,
               batch: []):
        """
        Update the model weights (thus training the model) using a batch of samples. Update is performed using the
        SARSA Bellman equation to compute the model target q-values.

        :param session: the session of tensorflow currently running
        :param batch: a batch of samples each one consisting in a tuple of observation current, action current, reward, observation next, action next, episode done flag and sample weight
        :return: the loss and its relative absolute error
        """
        # Generate a full pass-through mask for each example in the batch
        masks: numpy.ndarray = numpy.zeros((len(batch[0]), *self._agent_action_space_shape), dtype=float)
        # Unpack the batch into numpy arrays
        observations_current, actions_current, rewards, observations_next, actions_next, episode_done_flags, weights = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        # Generate a one-hot encoded version of the observations if space type is discrete
        if self._observation_space_type == SpaceType.discrete:
            observations_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observations_current.reshape(-1)]
            observations_next: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observations_next.reshape(-1)]
        # Get the q-values from the model at both current observations and next observations
        # Next observation is estimated by the target network
        q_values_current: numpy.ndarray = session.run(self._main_network_q_values_predictions,
                                                      feed_dict={
                                                          self._main_network_observations: observations_current,
                                                          self._main_network_mask: masks
                                                      })
        q_values_next: numpy.ndarray = session.run(self._target_network_q_values_predictions,
                                                   feed_dict={
                                                       self._target_network_observations: observations_next,
                                                       self._target_network_mask: masks
                                                   })
        # Apply Bellman equation with the SARSA update rule
        for sample_index in range(len(actions_current)):
            # Extract current sample values
            action_current = actions_current[sample_index]
            action_next = actions_next[sample_index]
            reward: float = rewards[sample_index]
            episode_done: bool = episode_done_flags[sample_index]
            # Note: only the immediate reward can be assigned at end of the episode, i.e. when next observation is None
            if episode_done:
                q_values_current[sample_index, action_current] = reward
            else:
                q_values_current[sample_index, action_current] = reward + self.discount_factor * q_values_next[sample_index, action_next]
        # Train the model and save the value of the loss and of the absolute error as well as the summary
        _, loss, absolute_error = session.run([self._optimizer, self._loss, self._absolute_error],
                                              feed_dict={
                                                    self._main_network_observations: observations_current,
                                                    self._q_values_targets: q_values_current,
                                                    self._loss_weights: weights,
                                                    self._main_network_mask: masks
                                               })
        # Return the loss and the absolute error
        return loss, absolute_error

    @property
    def warmup_steps(self) -> int:
        return self._buffer_capacity
