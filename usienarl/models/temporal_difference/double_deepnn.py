# Import packages

import tensorflow
import numpy
import enum

# Import required src

from usienarl import Model, SpaceType, Config
from usienarl.libs import SumTree


class Buffer:
    """
    TODO: summary
    Prioritized experience replay memory. It uses a sum tree to store not only the samples but also the priority of
    the samples, where the priority of the samples is a representation of the probability of a sample.

    When adding a new sample, the default priority value of the new node associated with such sample is set to
    the maximum defined priority. This value is iteratively changed by the algorithm when update is called.

    When getting the samples, a priority segment inversely proportional to the amount of required samples is generated.
    Samples are then uniformly taken for that segment, and stored in a minibatch which is returned.
    Also, a weight to compensate for the over-presence of higher priority samples is returned by the get_sample method,
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
              observation_current, action, reward: float, observation_next):
        # Find the current max priority on the tree leafs
        max_priority: float = numpy.max(self._sum_tree.leafs)
        # If the max priority is zero set it to the minimum defined
        if max_priority <= 0:
            max_priority = self._MINIMUM_ALLOWED_PRIORITY
        # Set the max priority as the default one for this new sample
        # Note: we set the max priority for each new sample and then improve on it iteratively during training
        self._sum_tree.add((observation_current, action, reward, observation_next), max_priority)

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
        # Return the sample (minibatch) with related weights
        return [numpy.array(observations_current), numpy.array(actions), numpy.array(rewards), numpy.array(observations_next), importance_sampling_weights]

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

class UpdateRule(enum.Enum):
    """
    TODO: summary
    """
    q_learning = 0
    sarsa = 1
    expected_sarsa = 2


class Estimator:
    """
    Estimator defining the real DDQN _model. It is used to define two identical models: target network and q-network.

    It is generated given the size of the observation and action spaces and the hidden layer config defining the
    hidden layers of the DDQN.
    """

    def __init__(self,
                 scope: str,
                 observation_space_shape, action_space_shape,
                 hidden_layers_config: Config):
        self.scope = scope
        with tensorflow.variable_scope(self.scope):
            # Define inputs of the estimator as a float adaptable array with shape Nx(S) where N is the number of examples and (S) the shape of the state
            self.inputs = tensorflow.placeholder(shape=[None, *observation_space_shape], dtype=tensorflow.float32, name="inputs")
            # Define the estimator network hidden layers from the config
            hidden_layers_output = hidden_layers_config.apply_hidden_layers(self.inputs)
            # Define outputs as an array of neurons of size NxA and with linear activation functions
            # Define the targets for learning with the same NxA adaptable size
            # Note: N is the number of examples and A the size of the action space (DDQN only supports discrete actions spaces)
            self.outputs = tensorflow.layers.dense(hidden_layers_output, *action_space_shape, name="outputs")
            self.targets = tensorflow.placeholder(shape=[None, *action_space_shape], dtype=tensorflow.float32, name="targets")
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


class DoubleDeepNN(Model):
    """
    DDQN (Double Deep Q-Network) _model. The _model is a deep neural network which hidden layers can be defined by a config
    parameter. It uses a target network and a q-network to correctly evaluate the expected future reward in order
    to stabilize learning.

    In order to synchronize the target network and the q-network, every some interval steps the weight have to be copied
    from the q-network to the target network.

    It is further enhanced by the update process of the q-value. In the DDQN the q-network estimates the outputs given
    the current state, but the best predicted action (the index of the best predicted output) is chosen by the target
    network.

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
                 hidden_layers_config: Config,
                 update_rule: UpdateRule = UpdateRule.q_learning):
        # Define deep-nn model attributes
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        # Define internal deep-nn model attributes
        self._update_rule: UpdateRule = update_rule
        self._buffer_capacity: int = buffer_capacity
        self._minimum_sample_probability: float = minimum_sample_probability
        self._random_sample_trade_off: float = random_sample_trade_off
        self._importance_sampling_value: float = importance_sampling_value
        self._importance_sampling_value_increment: float = importance_sampling_value_increment
        self._hidden_layers_config: Config = hidden_layers_config
        # Define deep-nn model empty attributes
        self.buffer: Buffer = None
        # Define internal deep-nn model empty attributes
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
        super(DoubleDeepNN, self).__init__(name)
        # Define the types of allowed observation and action spaces
        self._supported_observation_space_types.append(SpaceType.discrete)
        self._supported_observation_space_types.append(SpaceType.continuous)
        self._supported_action_space_types.append(SpaceType.discrete)

    def _define_graph(self):
        # Define two estimator, one for target network and one for main network, with identical structure
        self._main_network = Estimator(self._scope + "/" + self._name + "/MainNetwork",
                                       self._observation_space_shape, self._action_space_shape,
                                       self._hidden_layers_config)
        self._target_network = Estimator(self._scope + "/" + self._name + "/TargetNetwork",
                                         self._observation_space_shape, self._action_space_shape,
                                         self._hidden_layers_config)
        # Assign the main network properties to the model properties (main network is the actual model)
        self._main_network_inputs = self._main_network.inputs
        self._main_network_outputs = self._main_network.outputs
        self._targets = self._main_network.targets
        self._absolute_error = self._main_network.absolute_error
        self._loss = self._main_network.loss
        self._loss_weights = self._main_network.loss_weights
        # Assign the target network outputs and inputs to the specific target outputs/inputs of the model
        self._target_network_outputs = self._target_network.outputs
        self._target_network_inputs = self._target_network.inputs
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
            self.summary = tensorflow.summary.merge([tensorflow.summary.scalar("loss", self._loss)])

    def get_all_actions(self,
                        session,
                        observation_current):
        # Generate a one-hot encoded version of the observation
        observation_current_one_hot: numpy.ndarray = numpy.identity(*self._observation_space_shape)[observation_current]
        # Return all the predicted q-values given the current observation
        return session.run(self._main_network_outputs,
                           feed_dict={self._main_network_inputs: [observation_current_one_hot]})

    def get_best_action(self,
                        session,
                        observation_current):
        # Return the predicted action given the current observation
        return numpy.argmax(self.get_all_actions(session, observation_current))[0]

    def get_best_action_and_all_actions(self,
                                        session,
                                        observation_current):
        # Get all actions
        all_actions = self.get_all_actions(session, observation_current)
        # Return the best action and all the actions
        return numpy.argmax(all_actions)[0], all_actions

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
        observations_current, actions, rewards, observations_next, weights = batch[0], batch[1], batch[2], batch[3], batch[4]
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
        q_values_current: numpy.ndarray = session.run(self._main_network_outputs, feed_dict={self._main_network_inputs: observations_current_input})
        q_values_next_main_network: numpy.ndarray = session.run(self._main_network_outputs, feed_dict={self._main_network_inputs: observations_next_input})
        q_values_next_target_network: numpy.ndarray = session.run(self._target_network_outputs, feed_dict={self._target_network_inputs: observations_next_input})
        # Apply Bellman equation with the required update rule (Q-Learning, SARSA or Expected SARSA)
        if self._update_rule == UpdateRule.q_learning:
            self._q_learning_update_rule(len(batch), observations_next, actions, q_values_current, q_values_next_target_network, rewards, self.discount_factor)
        elif self._update_rule == UpdateRule.sarsa:
            self._sarsa_update_rule(len(batch), observations_next, actions, q_values_current, q_values_next_target_network, rewards, self.discount_factor)
        else:
            self._expected_sarsa_update_rule(len(batch), observations_next, actions, q_values_current, q_values_next_target_network, rewards, self.discount_factor)
        # Train the model and save the value of the loss and of the absolute error as well as the summary
        _, loss, absolute_error, summary = session.run([self._optimizer, self._loss, self._absolute_error, self._summary],
                                                       feed_dict={
                                                            self._main_network_inputs: observations_current_input,
                                                            self._targets: q_values_current,
                                                            self._loss_weights: weights
                                                       })
        # Return the loss, the absolute error and relative summary
        return summary, loss, absolute_error






    def update_batch(self,
                     session,
                     episode: int, episodes: int, step: int,
                     batch: [], sample_weights: []):
        """
        Overridden method of QLearningModel class: check its docstring for further information.
        """
        # Copy the weight of the q-network in the target network, if required at this episode/step
        self._copy_weight(session, step)
        # Get the outputs depending on the type of space (discrete is one-hot encoded)
        if self._observation_space_type == SpaceType.discrete:
            # Get all current states in the batch
            states_current = numpy.array([numpy.identity(*self._observation_space_shape)[val[0]] for val in batch])
            # Get all next states in the batch (if equals to None, it means end of episode, and as such no next state)
            states_next = numpy.array([numpy.identity(*self._observation_space_shape)[val[3] if val[3] is not None else 0]
                                       for val in batch])
        else:
            # Get all current states in the batch
            states_current = numpy.array([val[0] for val in batch])
            # Get all next states in the batch (if equals to None, it means end of episode, and as such no next state)
            states_next = numpy.array([val[3] if val[3] is not None
                                       else numpy.zeros(self._observation_space_shape, dtype=float) for val in
                                       batch])
        # Get the outputs at the current states and at the next states
        # Note: next states are computed by both the q-network and the target network
        outputs_current = session.run(self.outputs,
                                      feed_dict={self._inputs: states_current})
        outputs_next_q_network = session.run(self.outputs,
                                             feed_dict={self._inputs: states_next})
        outputs_next_target_network = session.run(self._outputs_target,
                                                  feed_dict={self._inputs_target: states_next})
        # Define training arrays
        inputs = numpy.zeros((len(batch), *self._observation_space_shape))
        targets = numpy.zeros((len(batch), *self._action_space_shape))
        for i, example in enumerate(batch):
            state_current, action, reward, state_next = example[0], example[1], example[2], example[3]
            # Apply Bellman equation, modifying the weights at the current states with the discounted future reward of
            # the next states given the actions
            if state_next is None:
                # Only the immediate reward can be assigned at end of the episode
                outputs_current[i, action] = reward
            else:
                # Predict the output using the q-network and then get the q-value estimated by the target network at the
                # same index
                predicted_output_index: int = numpy.argmax(outputs_next_q_network[i])
                outputs_current[i, action] = reward + self.discount_factor * outputs_next_target_network[i][predicted_output_index]
            # Insert training data in training arrays depending on the observation space type
            if self._observation_space_type == SpaceType.discrete:
                inputs[i] = numpy.identity(*self._observation_space_shape)[state_current]
            else:
                inputs[i] = state_current
            # The current outputs modified by the Bellman equation are now used as target for the _model
            targets[i] = outputs_current[i]
        # Feed the training arrays into the network and run the optimizer while also evaluating new weights values
        # Save the value of the loss and of the absolute error as well as the _summary
        _, loss, absolute_error, summary = session.run([self._optimizer, self._loss, self._absolute_error, self.summary],
                                                       feed_dict={
                                                       self._inputs: inputs,
                                                       self._targets: targets,
                                                       self._loss_weights: sample_weights
                                                       })
        # Return the loss, the absolute error and the _summary
        return loss, absolute_error, summary

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

    @staticmethod
    def _q_learning_update_rule(batch_size: int,
                                observations_next, actions,
                                q_values_current: numpy.ndarray,
                                q_values_next_main_network: numpy.ndarray, q_values_next_target_network: numpy.ndarray,
                                rewards, discount_factor: float):
        """
        Update the Q-Values target to be estimated by the _model using q-learning update rule for the Bellman equation:
        Q(s, a) = R + gamma * max_a(Q(s')) where the index of the max is computed by the main network.

        :param observations_next: the next observation for each sample
        :param actions: the actions taken by the agent at each sample
        :param q_values_current: the q-values output for the current observation (the target to be updated)
        :param q_values_next_target_network: the q-values output for the next observation as predicted by the main network (used to update the target)
        :param q_values_next_target_network: the q-values output for the next observation as predicted by the target network (used to update the target)
        :param rewards: the rewards obtained by the agent at each sample
        :param discount_factor: the discount factor set for the _model, i.e. gamma
        """
        for sample_index in range(batch_size):
            # Extract current sample values
            observation_next = observations_next[sample_index]
            action = actions[sample_index]
            reward: float = rewards[sample_index]
            # Update the q-values for current observation using Q-Learning Bellman equation
            # Note: only the immediate reward can be assigned at end of the episode, i.e. when next observation is None
            if observation_next is None:
                q_values_current[sample_index, action] = reward
            else:
                # Predict the best action index with the main network and execute the update with the target network
                best_action_index: int = numpy.argmax(q_values_next_main_network[sample_index])
                q_values_current[sample_index, action] = rewards + discount_factor * q_values_next_target_network[sample_index, best_action_index]

    @staticmethod
    def _sarsa_update_rule(batch_size: int,
                           observations_next, actions,
                           q_values_current: numpy.ndarray,
                           q_values_next_main_network: numpy.ndarray, q_values_next_target_network: numpy.ndarray,
                           rewards, discount_factor: float):
        """
        Update the Q-Values target to be estimated by the _model using SARSA update rule for the Bellman equation:
        Q(s, a) = R + gamma * Q(s', a)

        :param observations_next: the next observation for each sample
        :param actions: the actions taken by the agent at each sample
        :param q_values_current: the q-values output for the current observation (the target to be updated)
        :param q_values_next_target_network: the q-values output for the next observation as predicted by the main network (used to update the target)
        :param q_values_next_target_network: the q-values output for the next observation as predicted by the target network (used to update the target)
        :param rewards: the rewards obtained by the agent at each sample
        :param discount_factor: the discount factor set for the _model, i.e. gamma
        """
        for sample_index in range(batch_size):
            # Extract current sample values
            observation_next = observations_next[sample_index]
            action = actions[sample_index]
            reward: float = rewards[sample_index]
            # Update the q-values for current observation using SARSA Bellman equation
            # Note: only the immediate reward can be assigned at end of the episode, i.e. when next observation is None
            if observation_next is None:
                q_values_current[sample_index, action] = reward
            else:
                chosen_action: int = q_values_next_main_network[sample_index]
                q_values_current[sample_index, action] = rewards + discount_factor * q_values_next[sample_index, action]

    @staticmethod
    def _expected_sarsa_update_rule(batch_size: int,
                                    observations_next, actions,
                                    q_values_current: numpy.ndarray, q_values_next: numpy.ndarray,
                                    rewards, discount_factor: float):
        """
        Update the Q-Values target to be estimated by the _model using Expected SARSA update rule for the Bellman equation:
        Q(s, a) = R + gamma * mean_a(Q(s'))

        :param observations_next: the next observation for each sample
        :param actions: the actions taken by the agent at each sample
        :param q_values_current: the q-values output for the current observation (the target to be updated)
        :param q_values_next: the q-values output for the next observation (used to update the target)
        :param rewards: the rewards obtained by the agent at each sample
        :param discount_factor: the discount factor set for the _model, i.e. gamma
        """
        for sample_index in range(batch_size):
            # Extract current sample values
            observation_next = observations_next[sample_index]
            action = actions[sample_index]
            reward: float = rewards[sample_index]
            # Update the q-values for current observation using Expected SARSA Bellman equation
            # Note: only the immediate reward can be assigned at end of the episode, i.e. when next observation is None
            if observation_next is None:
                q_values_current[sample_index, action] = reward
            else:
                q_values_current[sample_index, action] = rewards + discount_factor * numpy.average(q_values_next[sample_index])

