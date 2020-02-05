# Import packages

import tensorflow
import numpy
import scipy.signal
import random

# Import required src

from usienarl import SpaceType, Config, Model
from usienarl.utils import softmax


class Buffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting with the environment,
    and using Generalized Advantage Estimation (GAE-Lambda) for calculating the advantages of state-action pairs.

    The buffer is dynamically resizable.

    The buffer contains list of states (or observations), actions (used as targets), values (computed by the value stream
    of the _model itself during prediction), advantages (computed by the buffer using GAE when a trajectory finishes and
    fed back up in the policy stream to drive the loss), rewards (used to compute rewards-to-go) and rewards-to-go
    (computed inside the buffer itself and used as weight for the targets action when training the value stream).
    """

    def __init__(self,
                 discount_factor: float, lambda_parameter: float):
        # Define buffer components
        self._observations: [] = []
        self._actions: [] = []
        self._advantages: [] = []
        self._rewards: [] = []
        self._rewards_to_go: [] = []
        self._values: [] = []
        # Define parameters
        self._discount_factor: float = discount_factor
        self._lambda_parameter: float = lambda_parameter
        # Define buffer pointer
        self._pointer: int = 0
        self._path_start_index: int = 0

    def store(self,
              observation, action, reward: float, value: float):
        """
        Store the time-step in the buffer.

        :param observation: the current observation to store in the buffer
        :param action: the last action to store in the buffer
        :param reward: the reward obtained from the action at the current state to store in the buffer
        :param value: the value of the state as estimated by the value stream of the model to store in the buffer
        """
        # Append all data and increase the pointer
        self._observations.append(observation)
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._pointer += 1

    def get(self) -> []:
        """
        Get all of the data from the buffer, with advantages appropriately normalized (shifted to have mean zero and
        standard deviation equals to one). Also reset pointers in the buffer and the lists composing the buffer.

        :return a list containing the ndarrays of: states, actions, advantages, rewards-to-go
        """
        # Get a numpy array on the advantage list
        advantages_array: numpy.ndarray = numpy.array(self._advantages)
        # Execute the advantage normalization trick
        # Note: make sure mean and std are not zero!
        global_sum: float = numpy.sum(advantages_array)
        advantage_mean: float = global_sum / advantages_array.size + 1e-8
        global_sum_squared: float = numpy.sum((advantages_array - advantage_mean) ** 2) + 1e-8
        advantage_std: float = numpy.sqrt(global_sum_squared / advantages_array.size) + 1e-8
        # Adjust advantages according to the trick
        advantages_array = ((advantages_array - advantage_mean) / advantage_std)
        # Save the necessary values as numpy arrays before reset
        states_array: numpy.ndarray = numpy.array(self._observations)
        actions_array: numpy.ndarray = numpy.array(self._actions)
        rewards_to_go_array: numpy.ndarray = numpy.array(self._rewards_to_go)
        # Reset the buffer and related pointers
        self._pointer = 0
        self._path_start_index = 0
        self._observations = []
        self._actions = []
        self._advantages = []
        self._rewards = []
        self._rewards_to_go = []
        self._values = []
        # Return all the buffer components
        return [states_array, actions_array, advantages_array, rewards_to_go_array]

    def finish_path(self,
                    value: float = 0):
        """
        Finish the path at the end of a trajectory. This looks back in the buffer to where the trajectory started,
        and uses rewards and value estimates from the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as the targets for the value stream optimization.

        :param value: the last reward given by the environment or the last predicted value if last state is not terminal
        """
        path_slice = slice(self._path_start_index, self._pointer)
        rewards: numpy.ndarray = numpy.array(self._rewards[path_slice] + [value])
        values: numpy.ndarray = numpy.array(self._values[path_slice] + [value])
        # Compute GAE-Lambda advantage estimation (compute advantages using the value in the buffer taken from the model)
        deltas: numpy.ndarray = rewards[:-1] + self._discount_factor * values[1:] - values[:-1]
        self._advantages[path_slice] = self._discount_cumulative_sum(deltas, self._discount_factor * self._lambda_parameter).tolist()
        # Compute rewards-to-go
        self._rewards_to_go[path_slice] = (self._discount_cumulative_sum(rewards, self._discount_factor)[:-1]).tolist()
        self._path_start_index = self._pointer

    @property
    def size(self) -> int:
        """
        The size of the buffer at the current time (it is dynamic).

        :return: the integer size of the buffer.
        """
        return self._pointer

    @staticmethod
    def _discount_cumulative_sum(vector: numpy.ndarray, discount: float) -> numpy.ndarray:
        """
        Compute discounted cumulative sums of vectors.
        Credits to rllab.

        :param vector: the vector on which to compute cumulative discounted sum (e.g. [x0, x1, x2])
        :return the discounted cumulative sum (e.g. [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x3])
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], vector[::-1], axis=0)[::-1]


class VanillaPolicyGradient(Model):
    """
    Vanilla Policy Gradient with GAE (Generalized Advantage Estimation) buffer.
    The algorithm is on-policy and executes updates every a certain number of episodes. It is an actor-critic model.
    The model is constituted by two streams, the policy stream (the actor part of the model) adn the value stream (the
    critic part of the model).
    The first stream computes and optimizes the policy loss. To drive the policy loss the advantages for each
    observation in the batch is required to be estimated. They are estimated by the buffer using GAE.
    The second stream computes the values of the current observations of the states and optimizes such estimation.
    To drive this loss, the value estimated by the stream itself for each state in the batch is required to be estimated.
    This estimation is done using the rewards. This stream is what makes the reward differentiable. Value update is run
    for a certain number of epochs.
    The buffer stores all the trajectories up to the update point. Since each episode can contains different numbers of
    steps, the buffer is dynamically resizable. When updated, the entire buffer is taken as batch and the buffer is
    cleared. This is necessary since to update the policy only trajectories collected using the current policy can be
    used. To stabilize learning and avoid overfitting, each update is done using minibatches.

    This algorithm is very likely to converge to local minimum.

    Supported observation spaces:
        - discrete
        - continuous

    Supported action spaces:
        - discrete
        - continuous
    """

    def __init__(self,
                 name: str,
                 hidden_layers_config: Config,
                 discount_factor: float = 0.99,
                 learning_rate_policy: float = 3e-4, learning_rate_value: float = 1e-3,
                 value_update_epochs: int = 80,
                 minibatch_size: int = 32,
                 lambda_parameter: float = 0.97):
        # Define model attributes
        self.learning_rate_policy: float = learning_rate_policy
        self.learning_rate_value: float = learning_rate_value
        self.discount_factor: float = discount_factor
        # Define internal model attributes
        self._hidden_layers_config: Config = hidden_layers_config
        self._value_update_epochs: int = value_update_epochs
        self._minibatch_size: int = minibatch_size
        self._lambda_parameter: float = lambda_parameter
        # Define vanilla policy gradient empty attributes
        self.buffer: Buffer = None
        # Define internal vanilla policy gradient empty attributes
        self._observations = None
        self._actions = None
        self._actions = None
        self._advantages = None
        self._rewards = None
        self._mask = None
        self._logits = None
        self._masked_logits = None
        self._expected_value = None
        self._std = None
        self._log_std = None
        self._log_likelihood_actions = None
        self._log_likelihood_predictions = None
        self._value_predicted = None
        self._value_stream_loss = None
        self._policy_stream_loss = None
        self._value_stream_optimizer = None
        self._policy_stream_optimizer = None
        self._approximated_entropy = None
        # Generate the base model
        super(VanillaPolicyGradient, self).__init__(name)
        # Define the types of allowed observation and action spaces
        self._supported_observation_space_types.append(SpaceType.discrete)
        self._supported_observation_space_types.append(SpaceType.continuous)
        self._supported_action_space_types.append(SpaceType.discrete)
        self._supported_action_space_types.append(SpaceType.continuous)

    def _define_graph(self):
        # Set the GAE buffer for the vanilla policy gradient algorithm
        self.buffer: Buffer = Buffer(self.discount_factor, self._lambda_parameter)
        # Define the tensorflow model
        with tensorflow.variable_scope(self._scope + "/" + self._name):
            # Define observations placeholder as an adaptable vector with shape Nx(O) where N is the number of examples and (O) the shape of the observation space
            # Note: it is the input of the model
            self._observations = tensorflow.placeholder(shape=[None, *self._observation_space_shape], dtype=tensorflow.float32, name="observations")
            # Define the actions placeholder as an adaptable vector with shape Nx(A) where N is the number of examples and (A) the shape of the action space
            self._actions = tensorflow.placeholder(shape=(None, *self._agent_action_space_shape), dtype=tensorflow.float32, name="targets")
            # Define the rewards placeholder as an adaptable vector of floats (they are actually rewards-to-go computed in the buffer)
            # Note: the model gets the rewards from the buffer
            self._rewards = tensorflow.placeholder(shape=(None,), dtype=tensorflow.float32, name="rewards")
            # Define the advantages placeholder as an adaptable vector of floats (computed with GAE in the buffer)
            # Note: the model gets the advantages from the buffer once computed using GAE on the values
            self._advantages = tensorflow.placeholder(shape=(None,), dtype=tensorflow.float32, name="advantages")
            # Define the policy stream
            # Note: this define the actor part of the model and its proper output (the predicted actions)
            with tensorflow.variable_scope("policy_stream"):
                # Define the policy stream network hidden layers from the config
                policy_stream_hidden_layers_output = self._hidden_layers_config.apply_hidden_layers(self._observations)
                # Change the model definition according to its action space type
                if self._agent_action_space_type == SpaceType.discrete:
                    # Define the mask placeholder
                    self._mask = tensorflow.placeholder(shape=(None, *self._agent_action_space_shape), dtype=tensorflow.float32, name="mask")
                    # Define the logits as outputs with shape NxA where N is the size of the batch, A is the action size when its type is discrete
                    self._logits = tensorflow.layers.dense(policy_stream_hidden_layers_output, *self._agent_action_space_shape, activation=None, kernel_initializer=tensorflow.contrib.layers.xavier_initializer(), name="logits")
                    # Compute the masked logits using the given additive mask
                    self._masked_logits = tensorflow.add(self._logits, self._mask)
                    # Define the predicted actions on the first shape dimension as a squeeze on the samples drawn from a categorical distribution over the logits
                    self._actions_predicted = tensorflow.squeeze(tensorflow.multinomial(logits=self._masked_logits, num_samples=1), axis=1)
                    # Define the log likelihood according to the categorical distribution on actions given and actions predicted
                    self._log_likelihood_actions, _ = self.get_categorical_log_likelihood(self._actions, self._logits, name="log_likelihood_actions")
                    self._log_likelihood_predictions, _ = self.get_categorical_log_likelihood(tensorflow.one_hot(self._actions_predicted, depth=self._agent_action_space_shape[0]), self._logits, name="log_likelihood_predictions")
                else:
                    # Define the expected value as the output of the deep neural network with shape Nx(A) where N is the number of inputs, (A) is the action shape when its type is continuous
                    self._expected_value = tensorflow.layers.dense(policy_stream_hidden_layers_output, *self._agent_action_space_shape, activation=None, kernel_initializer=tensorflow.contrib.layers.xavier_initializer(), name="expected_value")
                    # Define the log standard deviation and the standard deviation itself
                    self._log_std = tensorflow.get_variable(name="log_std", initializer=-0.5*numpy.ones(*self._agent_action_space_shape, dtype=numpy.float32))
                    self._std = tensorflow.exp(self._log_std, name="std")
                    # Define actions as the expected value summed up with a random gaussian vector multiplied by the standard deviation
                    self._actions_predicted = self._expected_value + tensorflow.random_normal(tensorflow.shape(self._expected_value)) * self._std
                    # Define the log likelihood according to the gaussian distribution on actions given and actions predicted
                    self._log_likelihood_actions = self.get_gaussian_log_likelihood(self._actions, self._expected_value, self._log_std, name="log_likelihood_actions")
                    self._log_likelihood_predictions = self.get_gaussian_log_likelihood(self._actions_predicted, self._expected_value, self._log_std, name="log_likelihood_predictions")
                # Define the policy stream loss as the mean of the advantages multiplied the log likelihood
                self._policy_stream_loss = -tensorflow.reduce_mean(self._advantages * self._log_likelihood_actions, name="policy_stream_loss")
                # Define the optimizer for the policy stream
                self._policy_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_policy).minimize(self._policy_stream_loss)
            # Define the value stream
            # Note: this is the critic part of the model and the system making the reward differentiable (the value computation)
            with tensorflow.variable_scope("value_stream"):
                # Define the value stream network hidden layers from the config and its output (a single float value)
                value_stream_hidden_layers_output = self._hidden_layers_config.apply_hidden_layers(self._observations)
                value_stream_output = tensorflow.layers.dense(value_stream_hidden_layers_output, 1, activation=None, kernel_initializer=tensorflow.contrib.layers.xavier_initializer(), name="value")
                # Define value by squeezing the output of the value stream
                self._value_predicted = tensorflow.squeeze(value_stream_output, axis=1, name="value_predicted")
                # Define value stream loss as the mean squared error of the difference between rewards-to-go given and the predicted value
                self._value_stream_loss = tensorflow.reduce_mean((self._rewards - self._value_predicted) ** 2, name="value_stream_loss")
                # Define the optimizer for the value stream
                self._value_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_value).minimize(self._value_stream_loss)
            # Define approximated entropy for the logger
            self._approximated_entropy = tensorflow.reduce_mean(-self._log_likelihood_actions, name="approximated_entropy")
            # Define the initializer
            self._initializer = tensorflow.global_variables_initializer()

    def sample_action(self,
                      session,
                      observation_current,
                      mask: numpy.ndarray = None):
        """
        Get the action sampled from the probability distribution of the model given the current observation and an optional mask.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :param mask: the optional mask used (only if the action space type is discrete) to remove certain actions from the prediction (-infinity to remove, 0.0 to pass-through)
        :return: the action predicted by the model, with the value estimated at the current state and the relative log-likelihood of the sampled action
        """
        # If there is no mask and the action space type is discrete generate a full pass-through mask
        if mask is None and self._agent_action_space_type == SpaceType.discrete:
            mask = numpy.zeros(self._agent_action_space_shape, dtype=float)
        # Return a random action sample given the current state and depending on the observation space type and also compute value estimate and log-likelihood
        if self._observation_space_type == SpaceType.discrete:
            # Generate a one-hot encoded version of the observation if observation space type is discrete
            observation_current_one_hot: numpy.ndarray = numpy.identity(*self._observation_space_shape)[observation_current]
            if self._agent_action_space_type == SpaceType.discrete:
                action, value, log_likelihood = session.run([self._actions_predicted, self._value_predicted, self._log_likelihood_predictions],
                                                            feed_dict={self._observations: [observation_current_one_hot], self._mask: [mask]})
            else:
                action, value, log_likelihood = session.run([self._actions_predicted, self._value_predicted, self._log_likelihood_predictions],
                                                            feed_dict={self._observations: [observation_current_one_hot]})
        else:
            if self._agent_action_space_type == SpaceType.discrete:
                action, value, log_likelihood = session.run([self._actions_predicted, self._value_predicted, self._log_likelihood_predictions],
                                                            feed_dict={self._observations: [observation_current], self._mask: [mask]})
            else:
                action, value, log_likelihood = session.run([self._actions_predicted, self._value_predicted, self._log_likelihood_predictions],
                                                            feed_dict={self._observations: [observation_current]})
        # Return the predicted action, the estimated value and the log-likelihood
        return action[0], value[0], log_likelihood[0]

    def get_value_and_log_likelihood(self,
                                     session,
                                     action,
                                     observation_current,
                                     mask: numpy.ndarray = None):
        """
        Get the estimated value of the given current observation and the log-likelihood of the given action.

        :param session: the session of tensorflow currently running
        :param action: the action of which to compute the log-likelihood
        :param observation_current: the current observation of the agent in the environment to estimate the value
        :param mask: the optional mask used (only if the action space type is discrete) to remove certain actions from the prediction (-infinity to remove, 0.0 to pass-through)
        :return: the value estimated at the current state and the log-likelihood of the given action
        """
        # If there is no mask and the action space type is discrete generate a full pass-through mask
        if mask is None and self._agent_action_space_type == SpaceType.discrete:
            mask = numpy.zeros(self._agent_action_space_shape, dtype=float)
        # Generate a one-hot encoded version of the action if action space type is discrete
        if self._agent_action_space_type == SpaceType.discrete:
            action: numpy.ndarray = numpy.eye(*self._agent_action_space_shape)[numpy.array(action).reshape(-1)]
        # Return the estimated value of the given current state and the log-likelihood of the given action
        if self._observation_space_type == SpaceType.discrete:
            # Generate a one-hot encoded version of the observation if observation space type is discrete
            observation_current_one_hot: numpy.ndarray = numpy.identity(*self._observation_space_shape)[observation_current]
            if self._agent_action_space_type == SpaceType.discrete:
                value, log_likelihood = session.run([self._value_predicted, self._log_likelihood_actions],
                                                    feed_dict={self._observations: [observation_current_one_hot],
                                                               self._mask: [mask],
                                                               self._actions: action})
            else:
                value, log_likelihood = session.run([self._value_predicted, self._log_likelihood_actions],
                                                    feed_dict={self._observations: [observation_current_one_hot],
                                                               self._actions: action})
        else:
            if self._agent_action_space_type == SpaceType.discrete:
                value, log_likelihood = session.run([self._value_predicted, self._log_likelihood_actions],
                                                    feed_dict={self._observations: [observation_current],
                                                               self._mask: [mask],
                                                               self._actions: action})
            else:
                value, log_likelihood = session.run([self._value_predicted, self._log_likelihood_actions],
                                                    feed_dict={self._observations: [observation_current],
                                                               self._actions: action})
        # Return the estimated value and the log-likelihood
        return value[0], log_likelihood[0]

    def get_value(self,
                  session,
                  observation_current):
        """
        Get the estimated value of the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to estimate the value
        :return: the value estimated at the current state
        """
        # Return the value predicted by the network at the current state
        if self._observation_space_type == SpaceType.discrete:
            # Generate a one-hot encoded version of the observation if observation space type is discrete
            observation_current_one_hot: numpy.ndarray = numpy.identity(*self._observation_space_shape)[observation_current]
            value = session.run(self._value_predicted,
                                feed_dict={self._observations: [observation_current_one_hot]})
        else:
            value = session.run(self._value_predicted,
                                feed_dict={self._observations: [observation_current]})
        # Return the estimated value
        return value[0]

    def get_log_likelihood(self,
                           session,
                           action,
                           observation_current,
                           mask: numpy.ndarray = None):
        """
        Get the the log-likelihood of the given action.

        :param session: the session of tensorflow currently running
        :param action: the action of which to compute the log-likelihood
        :param observation_current: the current observation of the agent in the environment
        :param mask: the optional mask used (only if the action space type is discrete) to remove certain actions from the prediction (-infinity to remove, 0.0 to pass-through)
        :return: the log-likelihood of the given action
        """
        # If there is no mask and the action space type is discrete generate a full pass-through mask
        if mask is None and self._agent_action_space_type == SpaceType.discrete:
            mask = numpy.zeros(self._agent_action_space_shape, dtype=float)
        # Generate a one-hot encoded version of the action if action space type is discrete
        if self._agent_action_space_type == SpaceType.discrete:
            action: numpy.ndarray = numpy.eye(*self._agent_action_space_shape)[numpy.array(action).reshape(-1)]
        # Return log likelihood over the given action with the given current observation and the given mask
        if self._observation_space_type == SpaceType.discrete:
            # Generate a one-hot encoded version of the observation if observation space type is discrete
            observation_current_one_hot: numpy.ndarray = numpy.identity(*self._observation_space_shape)[observation_current]
            if self._agent_action_space_type == SpaceType.discrete:
                log_likelihood = session.run(self._log_likelihood_actions,
                                             feed_dict={self._observations: [observation_current_one_hot],
                                                        self._mask: [mask],
                                                        self._actions: action})
            else:
                log_likelihood = session.run(self._log_likelihood_actions,
                                             feed_dict={self._observations: [observation_current_one_hot],
                                                        self._actions: action})
        else:
            if self._agent_action_space_type == SpaceType.discrete:
                log_likelihood = session.run(self._log_likelihood_actions,
                                             feed_dict={self._observations: [observation_current],
                                                        self._mask: [mask],
                                                        self._actions: action})
            else:
                log_likelihood = session.run(self._log_likelihood_actions,
                                             feed_dict={self._observations: [observation_current],
                                                        self._actions: action})
        # Return the log-likelihood
        return log_likelihood[0]

    def get_action_probabilities(self,
                                 session,
                                 observation_current,
                                 mask: numpy.ndarray = None) -> []:
        """
        Get all the action probabilities (softmax over masked logits if discrete, expected value and standard deviation if continuous) for the
        given current observation and an optional mask.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :param mask: the optional mask used (only if the action space type is discrete) to remove certain actions from the prediction (-infinity to remove, 0.0 to pass-through)
        :return: the list of action probabilities (softmax over masked logits or expected values and std wrapped in a list depending on the agent action space type)
        """
        if mask is None and self._agent_action_space_type == SpaceType.discrete:
            mask = numpy.zeros(self._agent_action_space_shape, dtype=float)
        # Get the logits or the expected value as the distribution of the action probabilities depending on the action space shape
        if self._agent_action_space_type == SpaceType.discrete:
            if self._observation_space_type == SpaceType.discrete:
                observation_current_one_hot: numpy.ndarray = numpy.identity(*self._observation_space_shape)[
                    observation_current]
                if self._agent_action_space_type == SpaceType.discrete:
                    logits = session.run([self._masked_logits],
                                         feed_dict={self._observations: [observation_current_one_hot],
                                                    self._mask: [mask]})
                else:
                    logits = session.run([self._masked_logits],
                                         feed_dict={self._observations: [observation_current_one_hot]})
            else:
                if self._agent_action_space_type == SpaceType.discrete:
                    logits = session.run([self._masked_logits],
                                         feed_dict={self._observations: [observation_current], self._mask: [mask]})
                else:
                    logits = session.run([self._masked_logits],
                                         feed_dict={self._observations: [observation_current]})
            # Return the softmax over the logits (probabilities of all actions)
            return softmax(logits[0]).flatten()
        else:
            expected_value, std = session.run([self._expected_value, self._std], feed_dict={self._observations: [observation_current]})
            # Return the expected value and the standard deviation wrapped in a list
            return [expected_value, std]

    def update(self,
               session,
               batch: []):
        # Unpack the batch
        observations, actions, advantages, rewards = batch[0], batch[1], batch[2], batch[3]
        # Generate a one-hot encoded version of the observations if observation space type is discrete
        if self._observation_space_type == SpaceType.discrete:
            observations: numpy.ndarray = numpy.eye(*self._observation_space_shape)[numpy.array(observations).reshape(-1)]
        # Generate a one-hot encoded version of the actions if action space type is discrete
        if self._agent_action_space_type == SpaceType.discrete:
            actions: numpy.ndarray = numpy.eye(*self._agent_action_space_shape)[numpy.array(actions).reshape(-1)]
        # Run the policy optimizer of the model in training mode, also compute policy stream loss and approximated entropy
        policy_update_minibatch_iterations: int = 0
        policy_stream_loss_total: float = 0.0
        approximated_entropy_total: float = 0.0
        for minibatch in self._get_minibatch(observations, actions, advantages, rewards, self._minibatch_size):
            # Unpack the minibatch
            minibatch_observations, minibatch_actions, minibatch_advantages, _ = minibatch
            # Update the policy
            _, policy_stream_loss, approximated_entropy = session.run([self._policy_stream_optimizer, self._policy_stream_loss,self._approximated_entropy],
                                                                      feed_dict={
                                                                            self._observations: minibatch_observations,
                                                                            self._actions: minibatch_actions,
                                                                            self._advantages: minibatch_advantages,
                                                                      })
            policy_update_minibatch_iterations += 1
            policy_stream_loss_total += policy_stream_loss
            approximated_entropy_total += approximated_entropy
        policy_stream_loss_average: float = policy_stream_loss_total / policy_update_minibatch_iterations
        approximated_entropy_average: float = approximated_entropy_total / policy_update_minibatch_iterations
        # Run the value optimizer of the model in training mode for the required amount of steps, also compute average value stream loss
        value_stream_loss_average: float = 0.0
        for _ in range(self._value_update_epochs):
            value_update_minibatch_iterations: int = 0
            value_stream_loss_total: float = 0.0
            for minibatch in self._get_minibatch(observations, actions, advantages, rewards, self._minibatch_size):
                # Unpack the minibatch
                minibatch_observations, _, minibatch_advantages, minibatch_rewards = minibatch
                # Update the value
                _, value_stream_loss = session.run([self._value_stream_optimizer, self._value_stream_loss],
                                                   feed_dict={
                                                       self._observations: minibatch_observations,
                                                       self._advantages: minibatch_advantages,
                                                       self._rewards: minibatch_rewards
                                                   })
                value_update_minibatch_iterations += 1
                value_stream_loss_total += value_stream_loss
            # The average is only really saved on the last value update to know it at the end of all the update steps
            value_stream_loss_average = value_stream_loss_total / value_update_minibatch_iterations
        # Generate the tensorflow summary
        summary = tensorflow.Summary()
        summary.value.add(tag="policy_stream_loss", simple_value=policy_stream_loss_average)
        summary.value.add(tag="value_stream_loss", simple_value=value_stream_loss_average)
        summary.value.add(tag="approximated_entropy", simple_value=approximated_entropy_average)
        # Return both losses and summary for the update sequence
        return summary, policy_stream_loss_average, value_stream_loss_average

    @property
    def warmup_steps(self) -> int:
        return 0

    @staticmethod
    def _get_minibatch(observations: [], actions: [], advantages: [], rewards: [],
                       minibatch_size: int) -> ():
        """
        Get a minibatch of the given minibatch size from the given batch (already unpacked).

        :param observations: the observations buffer in the batch
        :param actions: the actions buffer in the batch
        :param advantages: the advantages buffer in the batch
        :param rewards: the rewards buffer in the batch
        :param minibatch_size: the size of the minibatch
        :return: a tuple minibatch of shuffled samples of the given size
        """
        # Get a list of random ids of the batch
        batch_random_ids = random.sample(range(len(observations)), len(observations))
        # Generate the minibatches by shuffling the batch
        minibatch_observations, minibatch_actions, minibatch_advantages, minibatch_rewards = [], [], [], []
        for random_id in batch_random_ids:
            minibatch_observations.append(observations[random_id])
            minibatch_actions.append(actions[random_id])
            minibatch_advantages.append(advantages[random_id])
            minibatch_rewards.append(rewards[random_id])
            # Return the minibatch
            if len(minibatch_observations) % minibatch_size == 0:
                yield minibatch_observations, minibatch_actions, minibatch_advantages, minibatch_rewards
                # Clear the minibatch
                minibatch_observations, minibatch_actions, minibatch_advantages, minibatch_rewards = [], [], [], []

    @staticmethod
    def get_categorical_log_likelihood(actions_mask, logits, name: str):
        """
        Get log-likelihood for discrete action spaces (using a categorical distribution) on the given logits and with
        the actions mask.
        It uses tensorflow and as such should only be called in the define method.

        :param actions_mask: the actions used to mask the log-likelihood on the logits
        :param logits: the logits of the neural network
        :param name: the name of the tensorflow operation
        :return: the log-likelihood according to categorical distribution
        """
        # Define the unmasked likelihood as the log-softmax of the logits
        log_likelihood_unmasked = tensorflow.nn.log_softmax(logits)
        # Return the categorical log-likelihood by summing over the first axis of the target action mask multiplied
        # by the log-likelihood on the logits (unmasked, this is the masking operation) and the unmasked likelihood
        return tensorflow.reduce_sum(actions_mask * log_likelihood_unmasked, axis=1, name=name), log_likelihood_unmasked

    @staticmethod
    def get_gaussian_log_likelihood(actions, expected_value, log_std, name: str):
        """
        Get log-likelihood for continuous action spaces (using a gaussian distribution) on the given expected value and
        log-std and with the actions.
        It uses tensorflow and as such should only be called in the define method.

        :param actions: the actions used to compute the log-likelihood tensor
        :param expected_value: the expected value of the gaussian distribution
        :param log_std: the log-std of the gaussian distribution
        :param name: the name of the tensorflow operation
        :return: the log-likelihood according to gaussian distribution
        """
        # Define the log-likelihood tensor for the gaussian distribution on the given target actions
        log_likelihood_tensor = -0.5 * (((actions - expected_value) / (tensorflow.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + numpy.log(2 * numpy.pi))
        # Return the gaussian log-likelihood by summing over all the elements in the log-likelihood tensor defined above
        return tensorflow.reduce_sum(log_likelihood_tensor, axis=1, name=name)
