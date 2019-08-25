# Import packages

import tensorflow
import numpy
import scipy.signal

# Import required src

from usienarl import SpaceType, Config, Model


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
        self._states: [] = []
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
              state, action, reward: float, value: float):
        """
        Store the time-step in the buffer.

        :param state: the current state to store in the buffer
        :param action: the last action to store in the buffer
        :param reward: the reward obtained from the action at the current state to store in the buffer
        :param value: the value of the state as estimated by the value stream of the _model to store in the buffer
        """
        # Append all data and increase the pointer
        self._states.append(state)
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
        global_sum: float = numpy.sum(advantages_array)
        advantage_mean: float = global_sum / advantages_array.size
        global_sum_squared: float = numpy.sum((advantages_array - advantage_mean) ** 2)
        advantage_std: float = numpy.sqrt(global_sum_squared / advantages_array.size)
        # Adjust advantages according to the trick
        advantages_array = ((advantages_array - advantage_mean) / advantage_std)
        # Save the necessary values as numpy arrays before reset
        states_array: numpy.ndarray = numpy.array(self._states)
        actions_array: numpy.ndarray = numpy.array(self._actions)
        rewards_to_go_array: numpy.ndarray = numpy.array(self._rewards_to_go)
        # Reset the buffer and related pointers
        self._pointer = 0
        self._path_start_index = 0
        self._states = []
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
    Vanilla Policy Gradient with GAE (Generalized Advantage Estimation).
    The algorithm is on-policy and executes updates every a certain number of episodes.
    The model is constituted by two sub-models, or streams. The first stream computes and optimizes the policy loss,
    and to drive the loss the advantages for each state in the batch is required to be estimated. This stream is called
    policy stream.
    The second stream computes the value on the current states and optimizes such estimation. To drive the loss of such
    sub-model, the value estimated by the stream itself for each state in the batch is required to be estimated. This
    stream is called value stream.
    The advantage used to drive the policy stream loss is computed using GAE on the value estimated by the value stream
    and such computation is carried on in the buffer.

    The buffer stores all the trajectories up to the update point. Since each episode can contains different
    numbers of steps, the buffer is dynamically resizable.
    The algorithm is very likely to converge to local minima.

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
                 value_steps_for_update: int,
                 hidden_layers_config: Config,
                 lambda_parameter: float):
        # Define _model attributes
        self.learning_rate_policy: float = learning_rate_policy
        self.learning_rate_value: float = learning_rate_value
        self.discount_factor: float = discount_factor
        # Define internal _model attributes
        self._hidden_layers_config: Config = hidden_layers_config
        self._value_steps_for_update: int = value_steps_for_update
        self._lambda_parameter: float = lambda_parameter
        # Define vanilla policy gradient empty attributes
        self.buffer: Buffer = None
        # Define internal vanilla policy gradient empty attributes
        self._inputs = None
        self._actions = None
        self._targets = None
        self._advantages = None
        self._rewards = None
        self._mask = None
        self._logits = None
        self._expected_value = None
        self._std = None
        self._log_likelihood = None
        self._value = None
        self._value_stream_loss = None
        self._policy_stream_loss = None
        self._value_stream_optimizer = None
        self._policy_stream_optimizer = None
        # Generate the base _model
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
                self._logits = tensorflow.layers.dense(hidden_layers_output, *self._agent_action_space_shape, name="logits")
                # Compute the masked logits using the given mask
                masked_logits = tensorflow.add(self._logits, self._mask)
                # Define the actions on the first shape dimension as a squeeze on the samples drawn from a categorical distribution on the logits
                self._actions = tensorflow.squeeze(tensorflow.multinomial(logits=masked_logits, num_samples=1), axis=1)
                # Define the log likelihood according to the categorical distribution
                self._log_likelihood, _ = self.get_categorical_log_likelihood(self._targets, self._logits)
            else:
                # Define the expected value as the output of the deep neural network with shape Nx(A) where N is the number of inputs, (A) is the action shape
                self._expected_value = tensorflow.layers.dense(hidden_layers_output, *self._agent_action_space_shape, name="expected_value")
                # Define the log standard deviation
                log_std = tensorflow.get_variable(name="log_std", initializer=-0.5*numpy.ones(*self._agent_action_space_shape, dtype=numpy.float32))
                # Define the standard deviation
                self._std = tensorflow.exp(log_std, name="std")
                # Define actions as the expected value summed up with a noise vector multiplied by the standard deviation
                self._actions = self._expected_value + tensorflow.random_normal(tensorflow.shape(self._expected_value)) * self._std
                # Define the log likelihood according to the gaussian distribution
                self._log_likelihood = self.get_gaussian_log_likelihood(self._targets, self._expected_value, log_std)
            # Define the value estimator (a deep MLP)
            value_stream_hidden_layers_output = self._hidden_layers_config.apply_hidden_layers(self._inputs)
            value_stream_output = tensorflow.layers.dense(value_stream_hidden_layers_output, 1, activation=None)
            # Define value by squeezing the output of the advantage stream MLP
            self._value = tensorflow.squeeze(value_stream_output, axis=1, name="value")
            # Define the rewards as an adaptable vector of floats (they are actually rewards-to-go computed with GAE)
            self._rewards = tensorflow.placeholder(shape=(None, ), dtype=tensorflow.float32, name="rewards")
            # Define advantage loss as the mean squared error of the difference between computed rewards to go and the advantage
            self._value_stream_loss = tensorflow.reduce_mean((self._rewards - self._value) ** 2, name="value_loss")
            # Define the optimizer for the value stream (actually the MLP optimizer)
            self._value_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_value).minimize(self._value_stream_loss)
            # Define the advantages placeholder as an adaptable vector of floats (they are computed by the MLP and stored in the buffer)
            # Note: the model get the advantages from the buffer once computed using GAE on the values
            self._advantages = tensorflow.placeholder(shape=(None,), dtype=tensorflow.float32, name="advantages")
            # Define the loss as the mean of the rewards multiplied the log likelihood
            self._policy_stream_loss = -tensorflow.reduce_mean(self._advantages * self._log_likelihood, name="policy_loss")
            # Define the optimizer for the policy stream
            self._policy_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_policy).minimize(self._policy_stream_loss)
            # Define the _initializer
            self._initializer = tensorflow.global_variables_initializer()

    def _define_summary(self):
        with tensorflow.variable_scope(self._scope + "/" + self._name):
            # Define the _summary operation for this graph with losses summaries
            self.summary = tensorflow.summary.merge([tensorflow.summary.scalar("policy_stream_loss", self._policy_stream_loss),
                                                     tensorflow.summary.scalar("value_stream_loss", self._value_stream_loss)])

    def sample_action(self,
                      session,
                      observation_current,
                      mask: numpy.ndarray = None):
        """
        Get the action sampled from the probability distribution of the model given the current observation and an optional mask.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :param mask: the optional mask used (only if the action space type is discrete) to remove certain actions from the prediction (-infinity to remove, 0.0 to pass-through)
        :return: the action predicted by the model
        """
        # If there is no mask and the action space type is discrete generate a full pass-through mask
        if mask is None and self._agent_action_space_type == SpaceType.discrete:
            mask = numpy.zeros(self._agent_action_space_shape, dtype=float)
        # Return a random action sample given the current state and depending on the observation space type and also compute value estimate
        if self._observation_space_type == SpaceType.discrete:
            # Generate a one-hot encoded version of the observation if observation space type is discrete
            observation_current_one_hot: numpy.ndarray = numpy.identity(*self._observation_space_shape)[observation_current]
            if self._agent_action_space_type == SpaceType.discrete:
                action, value = session.run([self._actions, self._value],
                                            feed_dict={self._inputs: [observation_current_one_hot], self._mask: [mask]})
            else:
                action, value = session.run([self._actions, self._value],
                                            feed_dict={self._inputs: [observation_current_one_hot]})
        else:
            if self._agent_action_space_type == SpaceType.discrete:
                action, value = session.run([self._actions, self._value],
                                            feed_dict={self._inputs: [observation_current], self._mask: [mask]})
            else:
                action, value = session.run([self._actions, self._value],
                                            feed_dict={self._inputs: [observation_current]})
        # Return the predicted action and the estimated value
        return action[0], value[0]

    def get_action_probabilities(self,
                                 session,
                                 observation_current,
                                 mask: numpy.ndarray = None) -> []:
        """
        Get all the action probabilities (logits if discrete, expected value if continuous) for the given current observation and
        an optional mask.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon
        :param mask: the optional mask used (only if the action space type is discrete) to remove certain actions from the prediction (-infinity to remove, 0.0 to pass-through)
        :return: the list of action probabilities (logits or expected values depending on the agent action space type)
        """
        if mask is None and self._agent_action_space_type == SpaceType.discrete:
            mask = numpy.zeros(self._agent_action_space_shape, dtype=float)
        # Get the logits or the expected value as the distribution of the action probabilities depending on the action space shape
        if self._agent_action_space_type == SpaceType.discrete:
            if self._observation_space_type == SpaceType.discrete:
                observation_current_one_hot: numpy.ndarray = numpy.identity(*self._observation_space_shape)[observation_current]
                if self._agent_action_space_type == SpaceType.discrete:
                    logits = session.run([self._logits],
                                         feed_dict={self._inputs: [observation_current_one_hot], self._mask: [mask]})
                else:
                    logits = session.run([self._logits],
                                         feed_dict={self._inputs: [observation_current_one_hot]})
            else:
                if self._agent_action_space_type == SpaceType.discrete:
                    logits = session.run([self._logits],
                                         feed_dict={self._inputs: [observation_current]})
                else:
                    logits = session.run([self._logits],
                                         feed_dict={self._inputs: [observation_current], self._mask: [mask]})
            # Return the logits (probabilities of all actions)
            return logits
        else:
            expected_value, std = session.run([self._expected_value, self._std], feed_dict={self._inputs: [observation_current]})
            # Return the expected value and std of all actions
            return expected_value

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
        # Run the policy optimizer of the _model in training mode
        session.run(self._policy_stream_optimizer,
                    feed_dict={
                                self._inputs: observations,
                                self._targets: actions,
                                self._advantages: advantages
                              })
        # Run the value optimizer of the _model in training mode for the required amount of steps
        for _ in range(self._value_steps_for_update):
            session.run(self._value_stream_optimizer,
                        feed_dict={
                                    self._inputs: observations,
                                    self._advantages: advantages,
                                    self._rewards: rewards
                                  })
        # Compute the policy loss and value loss of the _model after this sequence of training and also compute the _summary
        policy_loss, value_loss, summary = session.run([self._policy_stream_loss, self._value_stream_loss, self.summary],
                                                       feed_dict={
                                                                   self._inputs: observations,
                                                                   self._targets: actions,
                                                                   self._rewards: rewards,
                                                                   self._advantages: advantages
                                                                 })
        # Return both losses and _summary for the update sequence
        return summary, policy_loss, value_loss

    @property
    def warmup_episodes(self) -> int:
        return 0

    @staticmethod
    def get_categorical_log_likelihood(target_actions_mask, logits):
        """
        Get log-likelihood for discrete action spaces (using a categorical distribution) on the given logits and with
        the target action mask.
        It uses tensorflow and as such should only be called in the _define method.

        :param target_actions_mask: the target actions used to mask the log-likelihood on the logits
        :param logits: the logits of the neural network
        :return: the log-likelihood according to categorical distribution
        """
        # Define the unmasked likelihood as the log-softmax of the logits
        log_likelihood_unmasked = tensorflow.nn.log_softmax(logits)
        # Return the categorical log-likelihood by summing over the first axis of the target action mask multiplied
        # by the log-likelihood on the logits (unmasked, this is the masking operation) and the unmasked likelihood
        return tensorflow.reduce_sum(target_actions_mask * log_likelihood_unmasked, axis=1, name="log_likelihood"), log_likelihood_unmasked

    @staticmethod
    def get_gaussian_log_likelihood(target_actions, expected_value, log_std):
        """
        Get log-likelihood for continuous action spaces (using a gaussian distribution) on the given expected value and
        log-std and with the target actions.
        It uses tensorflow and as such should only be called in the _define method.

        :param target_actions: the target actions used to compute the log-likelihood tensor
        :param expected_value: the expected value of the gaussian distribution
        :param log_std: the log-std of the gaussian distribution
        :return: the log-likelihood according to gaussian distribution
        """
        # Define the log-likelihood tensor for the gaussian distribution on the given target actions
        log_likelihood_tensor = -0.5 * (((target_actions - expected_value) / (tensorflow.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + numpy.log(2 * numpy.pi))
        # Return the gaussian log-likelihood by summing over all the elements in the log-likelihood tensor defined above
        return tensorflow.reduce_sum(log_likelihood_tensor, axis=1, name="log_likelihood")
