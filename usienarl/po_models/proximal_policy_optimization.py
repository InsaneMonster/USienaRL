# Import packages

import tensorflow
import numpy
import scipy.signal

# Import required src

from usienarl import SpaceType, Config, Model
from usienarl.utils.common import softmax


class Buffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting with the environment,
    and using Generalized Advantage Estimation (GAE-Lambda) for calculating the advantages of state-action pairs.

    The buffer is dynamically resizable.

    The buffer contains list of states (or observations), actions (used as targets), values (computed by the value stream
    of the _model itself during prediction), advantages (computed by the buffer using GAE when a trajectory finishes and
    fed back up in the policy stream to drive the loss), rewards (used to compute rewards-to-go), rewards-to-go
    (computed inside the buffer itself and used as weight for the targets action when training the value stream) and
    log-likelihoods (used to compute policy ratio).
    """

    def __init__(self,
                 discount_factor: float, lambda_parameter: float):
        # Define buffer components
        self._states: [] = []
        self._actions: [] = []
        self._advantages: [] = []
        self._rewards: [] = []
        self._rewards_to_go: [] = []
        self._values: [] = []
        self._log_likelihoods: [] = []
        # Define parameters
        self._discount_factor: float = discount_factor
        self._lambda_parameter: float = lambda_parameter
        # Define buffer pointer
        self._pointer: int = 0
        self._path_start_index: int = 0

    def store(self,
              state, action, reward: float, value: float, log_likelihood):
        """
        Store the time-step in the buffer.

        :param state: the current state to store in the buffer
        :param action: the last action to store in the buffer
        :param reward: the reward obtained from the action at the current state to store in the buffer
        :param value: the value of the state as estimated by the value stream of the model to store in the buffer
        :param log_likelihood: the log likelihood of the action on the state as estimated by the model
        """
        # Append all data and increase the pointer
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._log_likelihoods.append(log_likelihood)
        self._pointer += 1

    def get(self) -> []:
        """
        Get all of the data from the buffer, with advantages appropriately normalized (shifted to have mean zero and
        standard deviation equals to one). Also reset pointers in the buffer and the lists composing the buffer.

        :return a list containing the ndarrays of: states, actions, advantages, rewards-to-go, log-likelihoods
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
        states_array: numpy.ndarray = numpy.array(self._states)
        actions_array: numpy.ndarray = numpy.array(self._actions)
        rewards_to_go_array: numpy.ndarray = numpy.array(self._rewards_to_go)
        log_likelihoods_array: numpy.ndarray = numpy.array(self._log_likelihoods)
        # Reset the buffer and related pointers
        self._pointer = 0
        self._path_start_index = 0
        self._states = []
        self._actions = []
        self._advantages = []
        self._rewards = []
        self._rewards_to_go = []
        self._values = []
        self._log_likelihoods = []
        # Return all the buffer components
        return [states_array, actions_array, advantages_array, rewards_to_go_array, log_likelihoods_array]

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

    @staticmethod
    def _discount_cumulative_sum(vector: numpy.ndarray, discount: float) -> numpy.ndarray:
        """
        Compute discounted cumulative sums of vectors.
        Credits to rllab.

        :param vector: the vector on which to compute cumulative discounted sum (e.g. [x0, x1, x2])
        :return the discounted cumulative sum (e.g. [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x3])
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], vector[::-1], axis=0)[::-1]


class ProximalPolicyOptimization(Model):
    """
    Proximal Policy Optimization with GAE (Generalized Advantage Estimation).
    The algorithm is on-policy and executes updates every a certain number of episodes.
    The model is constituted by two sub-models, or streams. The first stream computes and optimizes the policy loss,
    and to drive the loss the advantages for each state in the batch is required to be estimated. The loss used an
    approximation of the KL divergence. This stream is called policy stream.
    The second stream computes the value on the current states and optimizes such estimation. To drive the loss of such
    sub-model, the value estimated by the stream itself for each state in the batch is required to be estimated. This
    stream is called value stream.
    The advantage used to drive the policy stream loss is computed using GAE on the value estimated by the value stream
    and such computation is carried on in the buffer.

    The buffer stores all the trajectories up to the update point. Since each episode can contains different
    numbers of steps, the buffer is dynamically resizable.
    The algorithm is very likely to converge to local minima but guarantees to not decrease its policy quality according
    to a minimum KL distance allowed between old and new policy.

    Supported observation spaces:
        - discrete
        - continuous

    Supported action spaces:
        - discrete
        - continuous
    """

    def __init__(self,
                 name: str,
                 discount_factor: float,
                 learning_rate_policy: float, learning_rate_value: float,
                 value_steps_for_update: int, policy_steps_for_update: int,
                 hidden_layers_config: Config,
                 lambda_parameter: float,
                 clip_ratio: float,
                 target_kl_divergence: float):
        # Define model attributes
        self.learning_rate_policy: float = learning_rate_policy
        self.learning_rate_value: float = learning_rate_value
        self.discount_factor: float = discount_factor
        # Define internal model attributes
        self._hidden_layers_config: Config = hidden_layers_config
        self._value_steps_for_update: int = value_steps_for_update
        self._policy_steps_for_update: int = policy_steps_for_update
        self._lambda_parameter: float = lambda_parameter
        self._clip_ratio: float = clip_ratio
        self._target_kl_divergence: float = target_kl_divergence
        # Define proximal policy optimization empty attributes
        self.buffer: Buffer = None
        # Define internal proximal policy optimization empty attributes
        self._inputs = None
        self._actions = None
        self._targets = None
        self._advantages = None
        self._rewards = None
        self._mask = None
        self._logits = None
        self._masked_logits = None
        self._expected_value = None
        self._std = None
        self._log_std = None
        self._log_likelihood_targets = None
        self._log_likelihood_actions = None
        self._previous_log_likelihoods = None
        self._value = None
        self._ratio = None
        self._min_advantage = None
        self._value_stream_loss = None
        self._policy_stream_loss = None
        self._value_stream_optimizer = None
        self._policy_stream_optimizer = None
        self._approximated_kl_divergence = None
        self._approximated_entropy = None
        self._clip_fraction = None
        # Generate the base model
        super(ProximalPolicyOptimization, self).__init__(name)
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
            # Define inputs of the estimator as a float adaptable array with shape Nx(S) where N is the number of examples and (S) the shape of the state
            self._inputs = tensorflow.placeholder(shape=[None, *self._observation_space_shape], dtype=tensorflow.float32, name="inputs")
            # Define the estimator network hidden layers from the config
            hidden_layers_output = self._hidden_layers_config.apply_hidden_layers(self._inputs)
            # Define the targets for learning with the same NxA adaptable size
            self._targets = tensorflow.placeholder(shape=(None, *self._agent_action_space_shape), dtype=tensorflow.float32, name="targets")
            # Change the _model definition according to its action space type
            if self._agent_action_space_type == SpaceType.discrete:
                # Define the mask placeholder
                self._mask = tensorflow.placeholder(shape=(None, *self._agent_action_space_shape), dtype=tensorflow.float32, name="mask")
                # Define the logits as outputs of the deep neural network with shape NxA where N is the number of inputs, A is the action size when its type is discrete
                self._logits = tensorflow.layers.dense(hidden_layers_output, *self._agent_action_space_shape, activation=None, kernel_initializer=tensorflow.contrib.layers.xavier_initializer(), name="logits")
                # Compute the masked logits using the given mask
                self._masked_logits = tensorflow.add(self._logits, self._mask)
                # Define the actions on the first shape dimension as a squeeze on the samples drawn from a categorical distribution on the logits
                self._actions = tensorflow.squeeze(tensorflow.multinomial(logits=self._masked_logits, num_samples=1), axis=1)
                # Define the log likelihood according to the categorical distribution on targets and actions
                self._log_likelihood_targets, _ = self.get_categorical_log_likelihood(self._targets, self._logits)
                self._log_likelihood_actions, _ = self.get_categorical_log_likelihood(tensorflow.one_hot(self._actions, depth=self._agent_action_space_shape[0]), self._logits)
            else:
                # Define the expected value as the output of the deep neural network with shape Nx(A) where N is the number of inputs, (A) is the action shape
                self._expected_value = tensorflow.layers.dense(hidden_layers_output, *self._agent_action_space_shape, activation=None, kernel_initializer=tensorflow.contrib.layers.xavier_initializer(), name="expected_value")
                # Define the log standard deviation
                self._log_std = tensorflow.get_variable(name="log_std", initializer=-0.5*numpy.ones(*self._agent_action_space_shape, dtype=numpy.float32))
                # Define the standard deviation
                self._std = tensorflow.exp(self._log_std, name="std")
                # Define actions as the expected value summed up with a noise vector multiplied by the standard deviation
                self._actions = self._expected_value + tensorflow.random_normal(tensorflow.shape(self._expected_value)) * self._std
                # Define the log likelihood according to the gaussian distribution on targets and actions
                self._log_likelihood_targets = self.get_gaussian_log_likelihood(self._targets, self._expected_value, self._log_std)
                self._log_likelihood_actions = self.get_gaussian_log_likelihood(self._actions, self._expected_value, self._log_std)
            # Define the value estimator (a deep MLP)
            value_stream_hidden_layers_output = self._hidden_layers_config.apply_hidden_layers(self._inputs)
            value_stream_output = tensorflow.layers.dense(value_stream_hidden_layers_output, 1, activation=None, kernel_initializer=tensorflow.contrib.layers.xavier_initializer(), name="value")
            # Define value by squeezing the output of the advantage stream MLP
            self._value = tensorflow.squeeze(value_stream_output, axis=1, name="value")
            # Define the rewards as an adaptable vector of floats (they are actually rewards-to-go computed with GAE)
            self._rewards = tensorflow.placeholder(shape=(None, ), dtype=tensorflow.float32, name="rewards")
            # Define advantage loss as the mean squared error of the difference between computed rewards to go and the advantage
            self._value_stream_loss = tensorflow.reduce_mean((self._rewards - self._value) ** 2, name="value_loss")
            # Define the optimizer for the value stream (actually the MLP optimizer)
            self._value_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_value).minimize(self._value_stream_loss)
            # Define the advantages placeholder as an adaptable vector of floats (they are stored in the buffer)
            # Note: the model get the advantages from the buffer once computed using GAE on the values
            self._advantages = tensorflow.placeholder(shape=(None,), dtype=tensorflow.float32, name="advantages")
            # Define the previous log-likelihood placeholder as an adaptable vector or float (they are stored in the buffer)
            # Note: the model get the previous log-likelihoods from the buffer
            self._previous_log_likelihoods = tensorflow.placeholder(shape=(None,), dtype=tensorflow.float32, name="previous_log_likelihood")
            # Define the ratio between the current log-likelihood and the previous one (when using exponential, minus is a division)
            self._ratio = tensorflow.exp(self._log_likelihood_targets - self._previous_log_likelihoods)
            # Define the minimum advantages with respect to clip ratio
            self._min_advantage = tensorflow.where(self._advantages > 0, (1 + self._clip_ratio) * self._advantages, (1 - self._clip_ratio) * self._advantages)
            # Define the policy stream loss as the mean of minimum between the advantages multiplied the ratio and the minimum advantage
            self._policy_stream_loss = -tensorflow.reduce_mean(tensorflow.minimum(self._ratio * self._advantages, self._min_advantage), name="policy_loss")
            # Define the optimizer for the policy stream
            self._policy_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_policy).minimize(self._policy_stream_loss)
            # Define approximated KL divergence (also used to early stop), approximated entropy and clip fraction for the logger
            self._approximated_kl_divergence = tensorflow.reduce_mean(self._previous_log_likelihoods - self._log_likelihood_targets, name="approximated_kl_divergence")
            self._approximated_entropy = tensorflow.reduce_mean(-self._log_likelihood_targets, name="approximated_entropy")
            self._clip_fraction = tensorflow.reduce_mean(tensorflow.cast(tensorflow.logical_or(self._ratio > (1 + self._clip_ratio), self._ratio < (1 - self._clip_ratio)), tensorflow.float32), name="clip_fraction")
            # Define the initializer
            self._initializer = tensorflow.global_variables_initializer()

    def _define_summary(self):
        with tensorflow.variable_scope(self._scope + "/" + self._name):
            # Define the _summary operation for this graph with losses and approximated KL and entropy summaries
            self.summary = tensorflow.summary.merge([tensorflow.summary.scalar("policy_stream_loss", self._policy_stream_loss),
                                                     tensorflow.summary.scalar("value_stream_loss", self._value_stream_loss),
                                                     tensorflow.summary.scalar("approximated_kl_divergence", self._approximated_kl_divergence),
                                                     tensorflow.summary.scalar("approximated_entropy", self._approximated_entropy),
                                                     tensorflow.summary.scalar("clip_fraction", self._clip_fraction)])

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
                action, value, log_likelihood = session.run([self._actions, self._value, self._log_likelihood_actions],
                                                            feed_dict={self._inputs: [observation_current_one_hot], self._mask: [mask]})
            else:
                action, value, log_likelihood = session.run([self._actions, self._value, self._log_likelihood_actions],
                                                            feed_dict={self._inputs: [observation_current_one_hot]})
        else:
            if self._agent_action_space_type == SpaceType.discrete:
                action, value, log_likelihood = session.run([self._actions, self._value, self._log_likelihood_actions],
                                                            feed_dict={self._inputs: [observation_current], self._mask: [mask]})
            else:
                action, value, log_likelihood = session.run([self._actions, self._value, self._log_likelihood_actions],
                                                            feed_dict={self._inputs: [observation_current]})
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
                value, log_likelihood = session.run([self._value, self._log_likelihood_targets],
                                                    feed_dict={self._inputs: [observation_current_one_hot],
                                                               self._mask: [mask],
                                                               self._targets: action})
            else:
                value, log_likelihood = session.run([self._value, self._log_likelihood_targets],
                                                    feed_dict={self._inputs: [observation_current_one_hot],
                                                               self._targets: action})
        else:
            if self._agent_action_space_type == SpaceType.discrete:
                value, log_likelihood = session.run([self._value, self._log_likelihood_targets],
                                                    feed_dict={self._inputs: [observation_current],
                                                               self._mask: [mask],
                                                               self._targets: action})
            else:
                value, log_likelihood = session.run([self._value, self._log_likelihood_targets],
                                                    eed_dict={self._inputs: [observation_current],
                                                              self._targets: action})
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
            value = session.run(self._value,
                                eed_dict={self._inputs: [observation_current_one_hot]})
        else:
            value = session.run(self._value,
                                feed_dict={self._inputs: [observation_current]})
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
                log_likelihood = session.run(self._log_likelihood_targets,
                                             feed_dict={self._inputs: [observation_current_one_hot],
                                                        self._mask: [mask],
                                                        self._targets: action})
            else:
                log_likelihood = session.run(self._log_likelihood_targets,
                                             feed_dict={self._inputs: [observation_current_one_hot],
                                                        self._targets: action})
        else:
            if self._agent_action_space_type == SpaceType.discrete:
                log_likelihood = session.run(self._log_likelihood_targets,
                                             feed_dict={self._inputs: [observation_current],
                                                        self._mask: [mask],
                                                        self._targets: action})
            else:
                log_likelihood = session.run(self._log_likelihood_targets,
                                             feed_dict={self._inputs: [observation_current],
                                                        self._targets: action})
        # Return the log-likelihood
        return log_likelihood[0]

    def get_action_probabilities(self,
                                 session,
                                 observation_current,
                                 mask: numpy.ndarray = None) -> []:
        """
        Get all the action probabilities (softmax over masked logits if discrete, expected value if continuous) for the
        given current observation and an optional mask.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :param mask: the optional mask used (only if the action space type is discrete) to remove certain actions from the prediction (-infinity to remove, 0.0 to pass-through)
        :return: the list of action probabilities (softmax over masked logits or expected values depending on the agent action space type)
        """
        if mask is None and self._agent_action_space_type == SpaceType.discrete:
            mask = numpy.zeros(self._agent_action_space_shape, dtype=float)
        # Get the logits or the expected value as the distribution of the action probabilities depending on the action space shape
        if self._agent_action_space_type == SpaceType.discrete:
            if self._observation_space_type == SpaceType.discrete:
                observation_current_one_hot: numpy.ndarray = numpy.identity(*self._observation_space_shape)[observation_current]
                if self._agent_action_space_type == SpaceType.discrete:
                    logits = session.run([self._masked_logits],
                                         feed_dict={self._inputs: [observation_current_one_hot], self._mask: [mask]})
                else:
                    logits = session.run([self._masked_logits],
                                         feed_dict={self._inputs: [observation_current_one_hot]})
            else:
                if self._agent_action_space_type == SpaceType.discrete:
                    logits = session.run([self._masked_logits],
                                         feed_dict={self._inputs: [observation_current], self._mask: [mask]})
                else:
                    logits = session.run([self._masked_logits],
                                         feed_dict={self._inputs: [observation_current]})
            # Return the softmax over the logits (probabilities of all actions)
            return softmax(logits[0]).flatten()
        else:
            expected_value, std = session.run([self._expected_value, self._std],
                                              feed_dict={self._inputs: [observation_current]})
            # Return the expected value
            return expected_value

    def update(self,
               session,
               batch: []):
        # Unpack the batch
        observations, actions, advantages, rewards, previous_log_likelihoods = batch[0], batch[1], batch[2], batch[3], batch[4]
        # Generate a one-hot encoded version of the observations if observation space type is discrete
        if self._observation_space_type == SpaceType.discrete:
            observations: numpy.ndarray = numpy.eye(*self._observation_space_shape)[numpy.array(observations).reshape(-1)]
        # Generate a one-hot encoded version of the actions if action space type is discrete
        if self._agent_action_space_type == SpaceType.discrete:
            actions: numpy.ndarray = numpy.eye(*self._agent_action_space_shape)[numpy.array(actions).reshape(-1)]
        # Run the policy optimizer of the model in training mode for the required amount of steps
        for _ in range(self._policy_steps_for_update):
            _, approximated_kl_divergence = session.run([self._policy_stream_optimizer, self._approximated_kl_divergence],
                                                        feed_dict={
                                                                    self._inputs: observations,
                                                                    self._targets: actions,
                                                                    self._advantages: advantages,
                                                                    self._previous_log_likelihoods: previous_log_likelihoods
                                                                  })
            # If approximated KL divergence is above a certain threshold stop updating the policy for now (early stop)
            if approximated_kl_divergence > 1.5 * self._target_kl_divergence:
                break
        # Run the value optimizer of the model in training mode for the required amount of steps
        for _ in range(self._value_steps_for_update):
            session.run(self._value_stream_optimizer,
                        feed_dict={
                                    self._inputs: observations,
                                    self._advantages: advantages,
                                    self._rewards: rewards
                                  })
        # Compute the policy loss and value loss of the model after this sequence of training and also compute the summary
        policy_loss, value_loss, summary = session.run([self._policy_stream_loss, self._value_stream_loss, self.summary],
                                                       feed_dict={
                                                                   self._inputs: observations,
                                                                   self._targets: actions,
                                                                   self._rewards: rewards,
                                                                   self._advantages: advantages,
                                                                   self._previous_log_likelihoods: previous_log_likelihoods
                                                                 })
        # Return both losses and summary for the update sequence
        return summary, policy_loss, value_loss

    @property
    def warmup_steps(self) -> int:
        return 0

    @staticmethod
    def get_categorical_log_likelihood(actions_mask, logits):
        """
        Get log-likelihood for discrete action spaces (using a categorical distribution) on the given logits and with
        the action mask.
        It uses tensorflow and as such should only be called in the define method.

        :param actions_mask: the actions used to mask the log-likelihood on the logits
        :param logits: the logits of the neural network
        :return: the log-likelihood according to categorical distribution
        """
        # Define the unmasked likelihood as the log-softmax of the logits
        log_likelihood_unmasked = tensorflow.nn.log_softmax(logits)
        # Return the categorical log-likelihood by summing over the first axis of the actions mask multiplied
        # by the log-likelihood on the logits (unmasked, this is the masking operation) and the unmasked likelihood
        return tensorflow.reduce_sum(actions_mask * log_likelihood_unmasked, axis=1, name="log_likelihood"), log_likelihood_unmasked

    @staticmethod
    def get_gaussian_log_likelihood(actions, expected_value, log_std):
        """
        Get log-likelihood for continuous action spaces (using a gaussian distribution) on the given expected value and
        log-std and with the actions.
        It uses tensorflow and as such should only be called in the define method.

        :param actions: the actions used to compute the log-likelihood tensor
        :param expected_value: the expected value of the gaussian distribution
        :param log_std: the log-std of the gaussian distribution
        :return: the log-likelihood according to gaussian distribution
        """
        # Define the log-likelihood tensor for the gaussian distribution on the given actions
        log_likelihood_tensor = -0.5 * (((actions - expected_value) / (tensorflow.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + numpy.log(2 * numpy.pi))
        # Return the gaussian log-likelihood by summing over all the elements in the log-likelihood tensor defined above
        return tensorflow.reduce_sum(log_likelihood_tensor, axis=1, name="log_likelihood")
