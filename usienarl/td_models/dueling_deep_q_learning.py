# Import packages

import tensorflow
import numpy

# Import required src

from usienarl import Model, SpaceType, Config
from usienarl.libs import SumTree


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
            last_steps.append((leaf_data[4]))
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


class Estimator:
    """
    Estimator defining the real dueling model. It is used to define two identical models: target network and q-network.

    It is generated given the size of the observation and action spaces and the hidden layer config defining the
    hidden layers of the dueling deep neural network.
    """

    def __init__(self,
                 scope: str,
                 observation_space_shape, agent_action_space_shape,
                 hidden_layers_config: Config):
        self.scope = scope
        with tensorflow.variable_scope(self.scope):
            # Define inputs of the estimator as a float adaptable array with shape Nx(S) where N is the number of examples and (S) the shape of the state
            self.inputs = tensorflow.placeholder(shape=[None, *observation_space_shape], dtype=tensorflow.float32, name="inputs")
            # Define the estimator network hidden layers from the config with two equal streams for value and advantage
            hidden_value_stream_output = hidden_layers_config.apply_hidden_layers(self.inputs)
            hidden_advantage_stream_output = hidden_layers_config.apply_hidden_layers(self.inputs)
            # Define the value and the advantage computation
            self._value = tensorflow.layers.dense(hidden_value_stream_output, 1, None, kernel_initializer=tensorflow.contrib.layers.xavier_initializer(), name="value")
            self._advantage = tensorflow.layers.dense(hidden_advantage_stream_output, *agent_action_space_shape, None, kernel_initializer=tensorflow.contrib.layers.xavier_initializer(), name="advantage")
            # Define outputs as an array of neurons of size Nx(A) where N is the number of examples and A the shape of the action space
            # and whose value is given by the following equation:
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a')) aka the aggregation layer
            self.outputs = self._value + tensorflow.subtract(self._advantage, tensorflow.reduce_mean(self._advantage, 1, True), name="outputs")
            # Define the targets for learning with the same Nx(A) adaptable size where N is the number of examples and A the shape of the action space
            self.targets = tensorflow.placeholder(shape=[None, *agent_action_space_shape], dtype=tensorflow.float32, name="targets")
            # Define the weights of the targets during the update process (e.g. the importance sampling weights)
            self.loss_weights = tensorflow.placeholder(shape=[None, 1], dtype=tensorflow.float32, name="loss_weights")
            # Define the absolute error
            self.absolute_error = tensorflow.abs(self.targets - self.outputs, name="absolute_error")
            # Define the estimator loss
            self.loss = tensorflow.reduce_mean(self.loss_weights * tensorflow.squared_difference(self.targets, self.outputs), name="loss")
            # Define the estimator weight parameters
            self.weight_parameters = [variable for variable
                                      in tensorflow.trainable_variables()
                                      if variable.name.startswith(self.scope)]
            self.weight_parameters = sorted(self.weight_parameters, key=lambda parameter: parameter.name)


class DuelingDeepQLearning(Model):
    """
    Dueling Double Deep Q-Learning (DDDQN) model with Q-Learning update rule.
    The model is a deep neural network which hidden layers can be defined by a config parameter.
    It uses a target network and a main network to correctly evaluate the expected future reward in order
    to stabilize learning.

    In order to synchronize the target network and the main network, every some interval steps the weight have to be
    copied from the main network to the target network.

    The update rule is the following (Bellman equation):
    Q(s, a) = R + gamma * max_a(Q(s')) where the max q-value action is estimated by the main network
    It uses the index of the max of the predicted q-value at the next state by the main network to compute an estimate
    with the target network used to update predicted q-value at current state.

    The DDDQN divide the output stream in advantage function and value function, making sure they are also identifiable
    (to make backpropagation still possible), and summing up to compute the outputs by the aggregation layer:
    Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))

    Supported observation spaces:
        - discrete
        - continuous

    Supported action spaces:
        - discrete
    """

    def __init__(self,
                 name: str,
                 learning_rate: float, discount_factor: float,
                 buffer_capacity: int,
                 minimum_sample_probability: float, random_sample_trade_off: float,
                 importance_sampling_value: float, importance_sampling_value_increment: float,
                 hidden_layers_config: Config):
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
        # Define model empty attributes
        self.buffer: Buffer = None
        # Define internal model empty attributes
        self._target_network: Estimator = None
        self._main_network: Estimator = None
        self._target_network_inputs = None
        self._target_network_outputs = None
        self._main_network_inputs = None
        self._main_network_outputs = None
        self._targets = None
        self._loss_weights = None
        self._absolute_error = None
        self._loss = None
        self._optimizer = None
        self._weight_copier = None
        # Generate the base model
        super(DuelingDeepQLearning, self).__init__(name)
        # Define the types of allowed observation and action spaces
        self._supported_observation_space_types.append(SpaceType.discrete)
        self._supported_observation_space_types.append(SpaceType.continuous)
        self._supported_action_space_types.append(SpaceType.discrete)

    def _define_graph(self):
        # Set the buffer
        self.buffer = Buffer(self._buffer_capacity, self._minimum_sample_probability, self._random_sample_trade_off,
                             self._importance_sampling_value, self._importance_sampling_value_increment)
        # Define two estimator, one for target network and one for main network, with identical structure
        self._main_network = Estimator(self._scope + "/" + self._name + "/MainNetwork",
                                       self._observation_space_shape, self._agent_action_space_shape,
                                       self._hidden_layers_config)
        self._target_network = Estimator(self._scope + "/" + self._name + "/TargetNetwork",
                                         self._observation_space_shape, self._agent_action_space_shape,
                                         self._hidden_layers_config)
        # Assign main and target networks to the model attributes
        self._main_network_inputs = self._main_network.inputs
        self._main_network_outputs = self._main_network.outputs
        self._target_network_outputs = self._target_network.outputs
        self._target_network_inputs = self._target_network.inputs
        self._targets = self._main_network.targets
        self._absolute_error = self._main_network.absolute_error
        self._loss = self._main_network.loss
        self._loss_weights = self._main_network.loss_weights
        # Define the optimizer
        self._optimizer = tensorflow.train.AdamOptimizer(self.learning_rate).minimize(self._loss)
        # Define the initializer
        self._initializer = tensorflow.global_variables_initializer()
        # Define the weight copy operation (to copy weights from main network to target network)
        self._weight_copier = []
        for main_network_parameter, target_network_parameter in zip(self._main_network.weight_parameters,
                                                                    self._target_network.weight_parameters):
            copy_operation = target_network_parameter.assign(main_network_parameter)
            self._weight_copier.append(copy_operation)

    def _define_summary(self):
        with tensorflow.variable_scope(self._scope + "/" + self._name):
            # Define the _summary operation for this graph with loss and absolute error summaries
            self._summary = tensorflow.summary.merge([tensorflow.summary.scalar("loss", self._loss)])

    def get_all_actions(self,
                        session,
                        observation_current):
        # Save by default the observation current input of the model to the given data
        observation_current_input = observation_current
        # Generate a one-hot encoded version of the observation if observation space is discrete
        if self._observation_space_type == SpaceType.discrete:
            observation_current_one_hot: numpy.ndarray = numpy.identity(*self._observation_space_shape)[observation_current]
            observation_current_input = observation_current_one_hot
        # Return the q-values predicted by the main network
        return session.run(self._main_network_outputs, feed_dict={self._main_network_inputs: [observation_current_input]})

    def get_best_action(self,
                        session,
                        observation_current):
        # Return the predicted action given the current observation
        return numpy.argmax(self.get_all_actions(session, observation_current))

    def get_best_action_and_all_actions(self,
                                        session,
                                        observation_current):
        # Get all actions
        all_actions = self.get_all_actions(session, observation_current)
        # Return the best action and all the actions
        return numpy.argmax(all_actions), all_actions

    def copy_weight(self,
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
        # Unpack the batch into numpy arrays
        observations_current, actions, rewards, observations_next, last_steps, weights = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        # Define the input observations to the model (to support both space types)
        observations_current_input = observations_current
        observations_next_input = observations_next
        # Generate a one-hot encoded version of the observations if space type is discrete
        if self._observation_space_type == SpaceType.discrete:
            observations_current_one_hot: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observations_current.reshape(-1)]
            observations_next_one_hot: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observations_next.reshape(-1)]
            # Set the model input to the one-hot encoded representation
            observations_current_input = observations_current_one_hot
            observations_next_input = observations_next_one_hot
        # Get the q-values from the model at both current observations and next observations
        # Next observation is estimated by the target network
        q_values_current: numpy.ndarray = session.run(self._main_network_outputs,
                                                      feed_dict={self._main_network_inputs: observations_current_input})
        q_values_next_main_network: numpy.ndarray = session.run(self._main_network_outputs,
                                                                feed_dict={self._main_network_inputs: observations_next_input})
        q_values_next_target_network: numpy.ndarray = session.run(self._target_network_outputs,
                                                                  feed_dict={self._target_network_inputs: observations_next_input})
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
                # Predict the best action index with the main network and execute the update with the target network
                best_action_index: int = numpy.argmax(q_values_next_main_network[sample_index])
                q_values_current[sample_index, action] = reward + self.discount_factor * q_values_next_target_network[sample_index, best_action_index]
        # Train the model and save the value of the loss and of the absolute error as well as the summary
        _, loss, absolute_error, summary = session.run([self._optimizer, self._loss, self._absolute_error, self._summary],
                                                       feed_dict={
                                                            self._main_network_inputs: observations_current_input,
                                                            self._targets: q_values_current,
                                                            self._loss_weights: weights
                                                       })
        # Return the loss, the absolute error and relative summary
        return summary, loss, absolute_error

    @property
    def warmup_episodes(self) -> int:
        return self._buffer_capacity

    @property
    def inputs_name(self) -> str:
        # Get the _name of the inputs of the tensorflow graph
        return "MainNetwork/inputs"

    @property
    def outputs_name(self) -> str:
        # Get the _name of the outputs of the tensorflow graph
        return "MainNetwork/outputs"
