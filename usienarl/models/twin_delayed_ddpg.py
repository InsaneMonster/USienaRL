# Import packages

import tensorflow
import numpy
import math

# Import required src

from usienarl import SpaceType, Config, Model


class Buffer:
    """
    A buffer for storing trajectories experienced by a TD3 agent interacting with the environment, it is a simple FIFO
    experience replay buffer.
    """
    def __init__(self,
                 capacity: int,
                 parallel_amount: int,
                 observation_space_shape: (),
                 action_space_shape: ()):
        # Define buffer attributes
        self._capacity: int = capacity
        self._parallel_amount: int = parallel_amount
        # Define buffer empty attributes
        self._pointer: int = 0
        self._size: int = 0
        self._observations_current: numpy.ndarray = numpy.zeros((self._capacity, *observation_space_shape), dtype=float)
        self._observations_next: numpy.ndarray = numpy.zeros((self._capacity, *observation_space_shape), dtype=float)
        self._actions: numpy.ndarray = numpy.zeros((self._capacity, *action_space_shape), dtype=float)
        self._rewards: numpy.ndarray = numpy.zeros(self._capacity, dtype=float)
        self._episode_done_flags: numpy.ndarray = numpy.zeros(self._capacity, dtype=float)
        self._episode_done_previous_step: numpy.ndarray = numpy.zeros(self._parallel_amount, dtype=bool)

    def store(self,
              observation_current: numpy.ndarray,
              action: numpy.ndarray,
              reward: numpy.ndarray,
              observation_next: numpy.ndarray,
              episode_done: numpy.ndarray):
        """
        Store the time-step in the buffer.

        :param observation_current: the current observation to store in the buffer
        :param action: the last action to store in the buffer
        :param reward: the reward obtained from the action at the current state to store in the buffer
        :param observation_next: the next observation to store in the buffer
        :param episode_done: whether or not this time-step was the last of the episode
        """
        # Serialize all the experiences to store in the buffer
        for i in range(self._parallel_amount):
            if self._episode_done_previous_step[i]:
                continue
            # Store data at the index targeted by the pointer
            self._observations_current[self._pointer] = observation_current[i]
            self._observations_next[self._pointer] = observation_next[i]
            self._actions[self._pointer] = action[i]
            self._rewards[self._pointer] = reward[i]
            self._episode_done_flags[self._pointer] = episode_done[i]
            # Update the pointer (make sure the first inserted elements are removed first when capacity is exceeded)
            self._pointer = (self._pointer + 1) % self._capacity
            # Update the size with respect to capacity
            self._size = min(self._size + 1, self._capacity)
        # Update the stored episode done flags
        self._episode_done_previous_step = episode_done.copy()

    def get(self,
            amount: int = 0) -> []:
        """
        Get a batch of data from the buffer of the given size. If size is not given all the buffer is used. The buffer
        is not cleared of the data being sampled.

        :param amount: the batch size of data to get
        :return a list containing the arrays of: current observations, actions, rewards, next observations and last step flags
        """
        # Adjust the amount with respect to the buffer current size
        if amount <= 0:
            amount = self._size
        # Return a set of random samples of as large as the given amount
        random_indexes: numpy.ndarray = numpy.random.randint(0, self._size, size=amount)
        return [self._observations_current[random_indexes], self._actions[random_indexes], self._rewards[random_indexes], self._observations_next[random_indexes], self._episode_done_flags[random_indexes]]

    def finish_trajectory(self):
        """
        Finish the trajectory, resetting episode done flags.
        """
        #  Reset stored episode done flags
        self._episode_done_previous_step: numpy.ndarray = numpy.zeros(self._parallel_amount, dtype=bool)

    @property
    def capacity(self) -> int:
        """
        The capacity of the buffer.

        :return: the integer capacity of the buffer
        """
        return self._capacity

    @property
    def size(self) -> int:
        """
        The size of the buffer at the current time.

        :return: the integer size of the buffer
        """
        return self._size


class Estimator:
    """
    Estimator defining the real Twin Delayed DDPG (TD3) model.  It is used to define two identical models:
    target network and main network.

    It is generated given the shape of the observation and action spaces and the hidden layer config defining the
    hidden layers of the network. Hidden layers can be defined separately for the actor network (the policy stream) and
    for the critic network (the q-stream).
    """

    def __init__(self,
                 scope: str,
                 observation_space_shape: (), agent_action_space_shape: (),
                 actor_hidden_layers_config: Config, critic_hidden_layers_config: Config,
                 reuse: bool = False, actions_placeholder=None):
        self.scope: str = scope
        with tensorflow.variable_scope(self.scope, reuse=reuse):
            # Define observations placeholder as a float adaptable array with shape Nx(O) where N is the number of examples and (O) the shape of the observation space
            self.observations = tensorflow.placeholder(shape=(None, *observation_space_shape), dtype=tensorflow.float32, name="observations")
            # If an action placeholder is not given generate a new one like in vanilla DDPG
            # Note: this is required to allow for target policy smoothing
            self.actions = actions_placeholder
            if self.actions is None:
                # Define the actions placeholder with adaptable size Nx(A) where N is the number of examples and (A) the shape of the action space
                self.actions = tensorflow.placeholder(shape=(None, *agent_action_space_shape), dtype=tensorflow.float32, name="actions")
            # Define the policy stream
            with tensorflow.variable_scope("policy_stream", reuse=reuse):
                # Define the boundaries placeholders to clip the actions
                self.lower_bound = tensorflow.placeholder(shape=(None, *agent_action_space_shape), dtype=tensorflow.float32, name="lower_bound")
                self.upper_bound = tensorflow.placeholder(shape=(None, *agent_action_space_shape), dtype=tensorflow.float32, name="upper_bound")
                # Define the estimator network policy stream hidden layers from the config
                policy_stream_hidden_layers_output = actor_hidden_layers_config.apply_hidden_layers(self.observations, reuse=reuse)
                # Define actions as an array of neurons with same adaptable size Nx(A) and with linear activation functions, clipped appropriately
                self.actions_predicted = tensorflow.clip_by_value(tensorflow.layers.dense(policy_stream_hidden_layers_output, *agent_action_space_shape, reuse=reuse), self.lower_bound, self.upper_bound, name="actions_predicted")
            # Define the q-streams (two for given actions and one for predicted actions)
            with tensorflow.variable_scope("q_stream_first", reuse=reuse):
                # Define the estimator network q-stream hidden layers from the config
                q_stream_targets_hidden_layers_output = critic_hidden_layers_config.apply_hidden_layers(tensorflow.concat([self.observations, self.actions], axis=-1, name="observations_actions_concat"), reuse=reuse)
                # Define the estimator network q-stream hidden layers from the config
                q_stream_actions_hidden_layers_output = critic_hidden_layers_config.apply_hidden_layers(tensorflow.concat([self.observations, self.actions_predicted], axis=-1, name="observations_predictions_concat"), reuse=True)
                # Define the estimator network predicted q-values over the predicted actions
                self.q_values_first_actions = tensorflow.squeeze(tensorflow.layers.dense(q_stream_targets_hidden_layers_output, 1, name="q_values_actions_head", reuse=reuse), axis=1, name="q_values_actions")
                # Define the estimator network predicted q-values over the given actions
                self.q_values_first_predictions = tensorflow.squeeze(tensorflow.layers.dense(q_stream_actions_hidden_layers_output, 1, name="q_values_predictions_head", reuse=reuse), axis=1, name="q_values_predictions")
            with tensorflow.variable_scope("q_stream_second", reuse=reuse):
                # Define the estimator network q-stream hidden layers from the config
                q_stream_targets_hidden_layers_output = critic_hidden_layers_config.apply_hidden_layers(tensorflow.concat([self.observations, self.actions], axis=-1, name="observations_actions_concat"), reuse=reuse)
                # Define the estimator network predicted q-values over the predicted actions
                self.q_values_second_actions = tensorflow.squeeze(tensorflow.layers.dense(q_stream_targets_hidden_layers_output, 1, name="q_values_actions_head", reuse=reuse), axis=1, name="q_values_actions")
            # Define the estimator weight parameters
            self.weight_parameters = [variable for variable in tensorflow.trainable_variables() if self.scope in variable.name]
            self.weight_parameters = sorted(self.weight_parameters, key=lambda parameter: parameter.name)


class TwinDelayedDDPG(Model):
    """
    Twin Delayed DDPG Model (TD3) with FIFO experience replay buffer.
    The model is a deep neural network which hidden layers can be defined by a config parameter. It is an actor-critic
    model. Hidden layers can be defined separately for the actor network (the policy stream) and for the critic
    network (the q-stream). It uses a target network and a main network to correctly evaluate the expected future
    reward in order to stabilize learning.

    In order to synchronize the target network and the main network, every some interval steps the weight have to be
    copied from the main network to the target network.

    TODO: is it good? add what it does

    Supported observation spaces:
        - discrete
        - continuous

    Supported action spaces:
        - continuous
    """

    def __init__(self,
                 name: str,
                 actor_hidden_layers_config: Config, critic_hidden_layers_config: Config,
                 learning_rate_policy: float = 1e-3, learning_rate_q_values: float = 1e-3,
                 discount_factor: float = 0.99, polyak_value: float = 0.995,
                 buffer_capacity: int = 1000000,
                 target_smoothing_noise: float = 0.2,
                 target_smoothing_noise_clip: float = 0.5,
                 error_clipping: bool = False,
                 huber_delta: float = 1.0):
        # Define attributes
        self.learning_rate_policy: float = learning_rate_policy
        self.learning_rate_q_values: float = learning_rate_q_values
        self.discount_factor: float = discount_factor
        self.polyak_value: float = polyak_value
        # Define internal attributes
        self._buffer_capacity: int = buffer_capacity
        self._actor_hidden_layers_config: Config = actor_hidden_layers_config
        self._critic_hidden_layers_config: Config = critic_hidden_layers_config
        self._target_smoothing_noise: float = target_smoothing_noise
        self._target_smoothing_noise_clip: float = target_smoothing_noise_clip
        self._error_clipping: bool = error_clipping
        self._huber_delta: float = huber_delta
        # Define empty attributes
        self.buffer: Buffer or None = None
        # Define internal empty attributes
        self._target_network: Estimator or None = None
        self._main_network: Estimator or None = None
        self._main_network_observations = None
        self._main_network_actions = None
        self._main_network_lower_bound = None
        self._main_network_upper_bound = None
        self._main_network_actions_predicted = None
        self._main_network_q_values_first_predictions = None
        self._main_network_q_values_first_actions = None
        self._main_network_q_values_second_actions = None
        self._target_network_observations = None
        self._target_network_lower_bound = None
        self._target_network_upper_bound = None
        self._target_network_q_values_first_predictions = None
        self._target_network_q_values_second_predictions = None
        self._target_network_min_q_values_predictions = None
        self._rewards = None
        self._episode_done_flags = None
        self._bellman_backup = None
        self._policy_stream_loss = None
        self._q_stream_first_absolute_error = None
        self._q_stream_second_absolute_error = None
        self._q_stream_first_loss = None
        self._q_stream_second_loss = None
        self._q_stream_loss = None
        self._policy_stream_optimizer = None
        self._q_stream_optimizer = None
        self._weight_copier = None
        self._weight_updater = None
        # Generate the base model
        super(TwinDelayedDDPG, self).__init__(name)
        # Define the types of allowed observation and action spaces
        self._supported_observation_space_types.append(SpaceType.discrete)
        self._supported_observation_space_types.append(SpaceType.continuous)
        self._supported_action_space_types.append(SpaceType.continuous)

    def _define_graph(self):
        # Set the buffer
        self.buffer = Buffer(self._buffer_capacity, self._parallel, self._observation_space_shape, self._agent_action_space_shape)
        # Define two estimator, one for target network and one for main network
        full_scope: str = self._scope + "/" + self._name
        self._main_network = Estimator(full_scope + "/MainNetwork",
                                       self._observation_space_shape, self._agent_action_space_shape,
                                       self._actor_hidden_layers_config, self._critic_hidden_layers_config)
        self._target_network = Estimator(full_scope + "/TargetNetwork",
                                         self._observation_space_shape, self._agent_action_space_shape,
                                         self._actor_hidden_layers_config, self._critic_hidden_layers_config)
        # Smooth the policy (actions predicted) of the target network to compute the q-values on them
        with tensorflow.variable_scope(full_scope + "/TargetNetwork", reuse=True):
            self._target_actions_epsilon = tensorflow.clip_by_value(tensorflow.random_normal(tensorflow.shape(self._target_network.actions_predicted), stddev=self._target_smoothing_noise), -self._target_smoothing_noise_clip, self._target_smoothing_noise_clip, name="actions_predicted_epsilon")
            self._target_actions_smoothed = tensorflow.clip_by_value(self._target_network.actions_predicted + self._target_actions_epsilon, self._target_network.lower_bound, self._target_network.upper_bound, name="actions_predicted_smoothed")
        # Reuse the target network using as new action placeholder the smoothed target actions predicted by the same target network
        self._target_network = Estimator(full_scope + "/TargetNetwork",
                                         self._observation_space_shape, self._agent_action_space_shape,
                                         self._actor_hidden_layers_config, self._critic_hidden_layers_config,
                                         reuse=True, actions_placeholder=self._target_actions_smoothed)
        # Assign main and target networks to the model attributes
        self._main_network_observations = self._main_network.observations
        self._main_network_actions = self._main_network.actions
        self._main_network_actions_predicted = self._main_network.actions_predicted
        self._main_network_q_values_first_predictions = self._main_network.q_values_first_predictions
        self._main_network_q_values_first_actions = self._main_network.q_values_first_actions
        self._main_network_q_values_second_actions = self._main_network.q_values_second_actions
        self._main_network_lower_bound = self._main_network.lower_bound
        self._main_network_upper_bound = self._main_network.upper_bound
        self._target_network_observations = self._target_network.observations
        self._target_network_q_values_first_predictions = self._target_network.q_values_first_actions
        self._target_network_q_values_second_predictions = self._target_network.q_values_second_actions
        self._target_network_lower_bound = self._target_network.lower_bound
        self._target_network_upper_bound = self._target_network.upper_bound
        # Define the shared part of the model
        with tensorflow.variable_scope(full_scope):
            # Define rewards and done flag placeholders as adaptable arrays with shape N where N is the number of examples
            self._rewards = tensorflow.placeholder(shape=(None,), dtype=tensorflow.float32, name="rewards")
            self._episode_done_flags = tensorflow.placeholder(shape=(None,), dtype=tensorflow.float32, name="episode_done_flags")
            # Compute the minimum q-values predicted by the target network
            self._target_network_min_q_values_predictions = tensorflow.minimum(self._target_network_q_values_first_predictions, self._target_network_q_values_second_predictions)
            # Bellman backup for the q-function
            self._bellman_backup = tensorflow.add(self._rewards, self.discount_factor * (1 - self._episode_done_flags) * self._target_network_min_q_values_predictions, name="bellman_backup")
            # Define policy stream loss
            self._policy_stream_loss = -tensorflow.reduce_mean(self._main_network_q_values_first_predictions, name="policy_stream_loss")
            # Define the absolute error over the q-stream loss
            self._q_stream_first_absolute_error = tensorflow.abs(self._main_network_q_values_first_actions - self._bellman_backup, name="q_stream_first_absolute_error")
            self._q_stream_second_absolute_error = tensorflow.abs(self._main_network_q_values_second_actions - self._bellman_backup, name="q_stream_second_absolute_error")
            # Define the q-stream loss with error clipping (huber loss) if required, otherwise mean square error loss
            # Note: define the loss as the sum of the losses of the two q-streams
            if self._error_clipping:
                self._q_stream_first_loss = tensorflow.reduce_mean(tensorflow.where(self._q_stream_first_absolute_error < self._huber_delta,
                                                                                    0.5 * tensorflow.square(self._q_stream_first_absolute_error),
                                                                                    self._q_stream_first_absolute_error - 0.5), name="q_stream_first_loss")
                self._q_stream_second_loss = tensorflow.reduce_mean(tensorflow.where(self._q_stream_second_absolute_error < self._huber_delta,
                                                                                     0.5 * tensorflow.square(self._q_stream_second_absolute_error),
                                                                                     self._q_stream_second_absolute_error - 0.5), name="q_stream_second_loss")
            else:
                self._q_stream_first_loss = tensorflow.reduce_mean(tensorflow.square(self._q_stream_first_absolute_error), name="q_stream_first_loss")
                self._q_stream_second_loss = tensorflow.reduce_mean(tensorflow.square(self._q_stream_second_absolute_error), name="q_stream_second_loss")
            self._q_stream_loss = tensorflow.add(self._q_stream_first_loss, self._q_stream_second_loss, name="q_stream_loss")
            # Define the optimizer
            self._policy_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_policy).minimize(self._policy_stream_loss, var_list=[x for x in tensorflow.global_variables() if full_scope + "/MainNetwork/policy_stream" in x.name])
            self._q_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_q_values).minimize(self._q_stream_loss, var_list=[x for x in tensorflow.global_variables() if full_scope + "/MainNetwork/q_stream" in x.name])
            # Define the initializer
            self._initializer = tensorflow.variables_initializer(tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES, full_scope), name="initializer")
            # Define the weight copy operation (to copy weights from main network to target network at the start)
            self._weight_copier = []
            for main_network_parameter, target_network_parameter in zip(self._main_network.weight_parameters, self._target_network.weight_parameters):
                copy_operation = target_network_parameter.assign(main_network_parameter)
                self._weight_copier.append(copy_operation)
            # Define the weight update operation (to update weight from main network to target network during training using Polyak averaging)
            self._weight_updater = tensorflow.group([tensorflow.assign(target_network_parameter, self.polyak_value * target_network_parameter + (1 - self.polyak_value) * main_network_parameter)
                                                    for main_network_parameter, target_network_parameter in zip(self._main_network.weight_parameters, self._target_network.weight_parameters)])

    def get_predicted_action_values(self,
                                    session,
                                    observation_current: numpy.ndarray,
                                    possible_actions: numpy.ndarray):
        """
        Get the predicted actions values (q-values) according to the model q-stream at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon, wrapped in a numpy array
        :param possible_actions: the optional list used to remove certain actions from the prediction (with discrete action space) or to bound the actions predicted (with continuous action space)
        :return: the predicted action values (q-values) predicted by the model
        """
        # Generate a one-hot encoded version of the observation if observation space is discrete
        if self._observation_space_type == SpaceType.discrete:
            observation_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[numpy.array(observation_current).reshape(-1)]
        # If there is no possible action list and the action space type is continuous generate an unbounded range, otherwise use the given range
        if possible_actions is None:
            lower_bound: numpy.ndarray = -math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
            upper_bound: numpy.ndarray = math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
        else:
            lower_bound: numpy.ndarray = possible_actions[:, 0]
            upper_bound: numpy.ndarray = possible_actions[:, 1]
        # Return the predicted q-values given the current observation
        return session.run(self._main_network_q_values_first_predictions,
                           feed_dict={
                               self._main_network_observations: observation_current,
                               self._main_network_lower_bound: lower_bound,
                               self._main_network_upper_bound: upper_bound
                           })

    def get_best_action(self,
                        session,
                        observation_current: numpy.ndarray,
                        possible_actions: numpy.ndarray):
        """
        Get the action predicted by the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon, wrapped in a numpy array
        :param possible_actions: the optional list used to remove certain actions from the prediction (with discrete action space) or to bound the actions predicted (with continuous action space)
        :return: the action predicted by the model
        """
        # Generate a one-hot encoded version of the observation if observation space is discrete
        if self._observation_space_type == SpaceType.discrete:
            observation_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[numpy.array(observation_current).reshape(-1)]
        # If there is no possible action list and the action space type is continuous generate an unbounded range, otherwise use the given range
        if possible_actions is None:
            lower_bound: numpy.ndarray = -math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
            upper_bound: numpy.ndarray = math.inf * numpy.ones((self._parallel, *self._agent_action_space_shape), dtype=float)
        else:
            lower_bound: numpy.ndarray = possible_actions[:, 0]
            upper_bound: numpy.ndarray = possible_actions[:, 1]
        # Return the predicted action given the current observation
        return session.run(self._main_network_actions_predicted,
                           feed_dict={
                               self._main_network_observations: observation_current,
                               self._main_network_lower_bound: lower_bound,
                               self._main_network_upper_bound: upper_bound
                           })

    def get_best_action_and_predicted_action_values(self,
                                                    session,
                                                    observation_current: numpy.ndarray,
                                                    possible_actions: numpy.ndarray):
        """
        Get the best action predicted by the model at the given current observation and the predicted action values
        (q-values) according to the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon, wrapped in a numpy array
        :param possible_actions: the optional list used to remove certain actions from the prediction (with discrete action space) or to bound the actions predicted (with continuous action space)
        :return: the best action predicted by the model and the action values (q-values) predicted by the model
        """
        # Return the best action and all predicted actions values
        return self.get_best_action(session, observation_current, possible_actions), self.get_predicted_action_values(session, observation_current, possible_actions)

    def weights_copy(self,
                     session):
        """
        Copy the weights from the main network to the target network.

        :param session: the session of tensorflow currently running
        """
        # Run all the weight copy operations
        session.run(self._weight_copier)

    def weights_update(self,
                       session):
        """
        Update the weights from the main network to the target network by Polyak averaging.

        :param session: the session of tensorflow currently running
        """
        # Run the weight updater
        session.run(self._weight_updater)

    def update_policy(self,
                      session,
                      batch: []):
        """
        Update the model weights (thus training the model) of the policy stream, given a batch of samples.

        :param session: the session of tensorflow currently running
        :param batch: a batch of samples each one consisting in a tuple of observation current, action, reward, observation next and episode done flags
        :return: the policy stream loss
        """
        # Unpack the batch into numpy arrays
        observations_current, actions, rewards, observations_next, episode_done_flags = batch[0], batch[1], batch[2], batch[3], batch[4]
        # Generate a one-hot encoded version of the observations if space type is discrete
        if self._observation_space_type == SpaceType.discrete:
            observations_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observations_current.reshape(-1)]
            observations_next: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observations_next.reshape(-1)]
        # Generate full pass-through boundaries
        lower_bound: numpy.ndarray = -math.inf * numpy.ones((len(rewards), *self._agent_action_space_shape), dtype=float)
        upper_bound: numpy.ndarray = math.inf * numpy.ones((len(rewards), *self._agent_action_space_shape), dtype=float)
        # Policy stream update
        _, policy_stream_loss = session.run([self._policy_stream_optimizer, self._policy_stream_loss],
                                            feed_dict={
                                                self._main_network_observations: observations_current,
                                                self._target_network_observations: observations_next,
                                                self._main_network_actions: actions,
                                                self._rewards: rewards,
                                                self._episode_done_flags: episode_done_flags,
                                                self._main_network_lower_bound: lower_bound,
                                                self._main_network_upper_bound: upper_bound,
                                                self._target_network_lower_bound: lower_bound,
                                                self._target_network_upper_bound: upper_bound
                                            })
        # Return policy stream loss
        return policy_stream_loss

    def update_q_values(self,
                        session,
                        batch: []):
        """
        Update the model weights (thus training the model) of the q-stream, given a batch of samples.

        :param session: the session of tensorflow currently running
        :param batch: a batch of samples each one consisting in a tuple of observation current, action, reward, observation next and episode done flags
        :return: the q-stream loss
        """
        # Unpack the batch into numpy arrays
        observations_current, actions, rewards, observations_next, episode_done_flags = batch[0], batch[1], batch[2], batch[3], batch[4]
        # Generate a one-hot encoded version of the observations if space type is discrete
        if self._observation_space_type == SpaceType.discrete:
            observations_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observations_current.reshape(-1)]
            observations_next: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observations_next.reshape(-1)]
        # Generate full pass-through boundaries
        lower_bound: numpy.ndarray = -math.inf * numpy.ones((len(rewards), *self._agent_action_space_shape), dtype=float)
        upper_bound: numpy.ndarray = math.inf * numpy.ones((len(rewards), *self._agent_action_space_shape), dtype=float)
        # Q-stream update
        _, q_stream_loss = session.run([self._q_stream_optimizer, self._q_stream_loss],
                                       feed_dict={
                                            self._main_network_observations: observations_current,
                                            self._target_network_observations: observations_next,
                                            self._main_network_actions: actions,
                                            self._rewards: rewards,
                                            self._episode_done_flags: episode_done_flags,
                                            self._main_network_lower_bound: lower_bound,
                                            self._main_network_upper_bound: upper_bound,
                                            self._target_network_lower_bound: lower_bound,
                                            self._target_network_upper_bound: upper_bound
                                       })
        # Return q-stream loss
        return q_stream_loss

    @property
    def warmup_steps(self) -> int:
        return self._buffer_capacity
