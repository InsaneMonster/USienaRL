# Import packages

import tensorflow
import numpy
import math

# Import required src

from usienarl import SpaceType, Config, Model


class Buffer:
    """
    A buffer for storing trajectories experienced by a DDPG agent interacting with the environment, it is a simple FIFO
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
    Estimator defining the real Deterministic Policy Gradient (DDPG) model. It is used to define two identical models:
    target network and main network.

    It is generated given the shape of the observation and action spaces and the hidden layer config defining the
    hidden layers of the network. Hidden layers can be defined separately for the actor network (the policy stream) and
    for the critic network (the q-stream).
    """

    def __init__(self,
                 scope: str,
                 observation_space_shape: (), agent_action_space_shape: (),
                 actor_hidden_layers_config: Config, critic_hidden_layers_config: Config):
        self.scope: str = scope
        with tensorflow.variable_scope(self.scope):
            # Define observations placeholder as a float adaptable array with shape Nx(O) where N is the number of examples and (O) the shape of the observation space
            self.observations = tensorflow.placeholder(shape=(None, *observation_space_shape), dtype=tensorflow.float32, name="observations")
            # Define the actions placeholder with adaptable size Nx(A) where N is the number of examples and (A) the shape of the action space
            self.actions = tensorflow.placeholder(shape=(None, *agent_action_space_shape), dtype=tensorflow.float32, name="actions")
            # Define the policy stream
            with tensorflow.variable_scope("policy_stream"):
                # Define the estimator network policy stream hidden layers from the config
                policy_stream_hidden_layers_output = actor_hidden_layers_config.apply_hidden_layers(self.observations)
                # Define actions as an array of neurons with same adaptable size Nx(A) and with linear activation functions, clipped appropriately
                self.actions_predicted = tensorflow.layers.dense(policy_stream_hidden_layers_output, *agent_action_space_shape, activation=tensorflow.nn.tanh, name="actions_predicted")
            # Define the q-stream (both for given actions and predicted actions)
            with tensorflow.variable_scope("q_stream"):
                # Define the estimator network q-stream hidden layers from the config for the target actions
                q_stream_actions_hidden_layers_output = critic_hidden_layers_config.apply_hidden_layers(tensorflow.concat([self.observations, self.actions], axis=-1))
                # Define the estimator network predicted q-values over the target actions
                self.q_values_actions = tensorflow.squeeze(tensorflow.layers.dense(q_stream_actions_hidden_layers_output, 1), axis=1, name="q_values_actions")
                # Define the estimator network q-stream hidden layers from the config for the predicted actions (automatically shared with targets' one)
                q_stream_predictions_hidden_layers_output = critic_hidden_layers_config.apply_hidden_layers(tensorflow.concat([self.observations, self.actions_predicted], axis=-1))
                # Define the estimator network predicted q-values over the predicted actions
                self.q_values_predictions = tensorflow.squeeze(tensorflow.layers.dense(q_stream_predictions_hidden_layers_output, 1), axis=1, name="q_values_predictions")
            # Define the estimator weight parameters
            self.weight_parameters = [variable for variable in tensorflow.trainable_variables() if self.scope in variable.name]
            self.weight_parameters = sorted(self.weight_parameters, key=lambda parameter: parameter.name)


class DeepDeterministicPolicyGradient(Model):
    """
    Deep Deterministic Policy Gradient Model (DDPG) with FIFO experience replay buffer.
    The model is a deep neural network which hidden layers can be defined by a config parameter. It is an actor-critic
    model. Hidden layers can be defined separately for the actor network (the policy stream) and for the critic
    network (the q-stream). It uses a target network and a main network to correctly evaluate the expected future
    reward in order to stabilize learning.

    In order to synchronize the target network and the main network, every some interval steps the weight have to be
    copied from the main network to the target network.

    This vanilla DDPG implementation tends to be very unstable and its performance varies greatly with respect to the
    hyperparameters. Usually the problem lies in the too high q-values estimations (going to +infinity with time) and
    it is a problem of the vanilla DDPG algorithm as a whole. To help stabilize learning, a SGD optimizer is used
    instead of Adan.

    Note: actions are predicted in the range [-1.0, 1.0] (by using tanh). A rescale is done with respect to lower and
    upper bound if a set of possible actions is given. If not given, the output will remain between [-1.0, 1.0].

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
                 error_clipping: bool = False,
                 huber_delta: float = 1.0):
        # Define model attributes
        self.learning_rate_policy: float = learning_rate_policy
        self.learning_rate_q_values: float = learning_rate_q_values
        self.discount_factor: float = discount_factor
        self.polyak_value: float = polyak_value
        # Define internal model attributes
        self._buffer_capacity: int = buffer_capacity
        self._actor_hidden_layers_config: Config = actor_hidden_layers_config
        self._critic_hidden_layers_config: Config = critic_hidden_layers_config
        self._error_clipping: bool = error_clipping
        self._huber_delta: float = huber_delta
        # Define model empty attributes
        self.buffer: Buffer or None = None
        # Define internal model empty attributes
        self._target_network: Estimator or None = None
        self._main_network: Estimator or None = None
        self._main_network_observations = None
        self._main_network_actions = None
        self._main_network_actions_predicted = None
        self._main_network_q_values_predictions = None
        self._main_network_q_values_actions = None
        self._target_network_observations = None
        self._target_network_q_values_predictions = None
        self._target_q_values = None
        self._policy_stream_loss = None
        self._q_stream_absolute_error = None
        self._q_stream_loss = None
        self._policy_stream_optimizer = None
        self._q_stream_optimizer = None
        self._weight_copier = None
        self._weight_updater = None
        # Generate the base model
        super(DeepDeterministicPolicyGradient, self).__init__(name)
        # Define the types of allowed observation and action spaces
        self._supported_observation_space_types.append(SpaceType.discrete)
        self._supported_observation_space_types.append(SpaceType.continuous)
        self._supported_action_space_types.append(SpaceType.continuous)

    def _define_graph(self):
        # Set the buffer
        self.buffer = Buffer(self._buffer_capacity, self._parallel, self._observation_space_shape, self._agent_action_space_shape)
        # Define two estimator, one for target network and one for main network, with identical structure
        full_scope: str = self._scope + "/" + self._name
        self._main_network = Estimator(full_scope + "/MainNetwork",
                                       self._observation_space_shape, self._agent_action_space_shape,
                                       self._actor_hidden_layers_config, self._critic_hidden_layers_config)
        self._target_network = Estimator(full_scope + "/TargetNetwork",
                                         self._observation_space_shape, self._agent_action_space_shape,
                                         self._actor_hidden_layers_config, self._critic_hidden_layers_config)
        # Assign main and target networks to the model attributes
        self._main_network_observations = self._main_network.observations
        self._main_network_actions = self._main_network.actions
        self._main_network_actions_predicted = self._main_network.actions_predicted
        self._main_network_q_values_predictions = self._main_network.q_values_predictions
        self._main_network_q_values_actions = self._main_network.q_values_actions
        self._target_network_observations = self._target_network.observations
        self._target_network_q_values_predictions = self._target_network.q_values_predictions
        # Define the shared part of the model
        with tensorflow.variable_scope(full_scope):
            # Define target q-values placeholders as adaptable arrays with shape N where N is the number of examples
            self._target_q_values = tensorflow.placeholder(shape=(None,), dtype=tensorflow.float32, name="target_q_values")
            # Define policy stream loss
            self._policy_stream_loss = -tensorflow.reduce_mean(self._main_network_q_values_predictions, name="policy_stream_loss")
            # Define the absolute error over the q-stream
            self._q_stream_absolute_error = tensorflow.abs(self._main_network_q_values_actions - self._target_q_values, name="q_stream_absolute_error")
            # Define the q-stream loss with error clipping (huber loss) if required, otherwise mean square error loss
            if self._error_clipping:
                # Compute the Huber loss
                self._q_stream_loss = tensorflow.reduce_mean(tensorflow.where(self._q_stream_absolute_error < self._huber_delta,
                                                                              0.5 * tensorflow.square(self._q_stream_absolute_error),
                                                                              self._q_stream_absolute_error - 0.5), name="q_stream_loss")
            else:
                # Compute MSE loss over the q-stream
                self._q_stream_loss = tensorflow.reduce_mean(tensorflow.square(self._q_stream_absolute_error), name="q_stream_loss")
            # Define the optimizer
            self._policy_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_policy).minimize(self._policy_stream_loss, var_list=[x for x in tensorflow.global_variables() if "MainNetwork/policy_stream" in x.name])
            self._q_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_q_values).minimize(self._q_stream_loss, var_list=[x for x in tensorflow.global_variables() if "MainNetwork/q_stream" in x.name])
            # Define the initializer
            self._initializer = tensorflow.variables_initializer(tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES, full_scope), name="initializer")
            # Define target network copy/update operations using main network weights
            self._weight_copier = []
            self._weight_updater = []
            for main_network_parameter, target_network_parameter in zip(self._main_network.weight_parameters, self._target_network.weight_parameters):
                # Define the weight copy operation (to copy weights from main network to target network at the start)
                copy_operation = tensorflow.assign(target_network_parameter, main_network_parameter)
                self._weight_copier.append(copy_operation)
                # Define the weight update operation (to update weight from main network to target network during training using Polyak averaging)
                update_operation = tensorflow.assign(target_network_parameter, self.polyak_value * target_network_parameter + (1 - self.polyak_value) * main_network_parameter)
                self._weight_updater.append(update_operation)

    def get_predicted_action_values(self,
                                    session,
                                    observation_current: numpy.ndarray):
        """
        Get the predicted actions values (q-values) according to the model q-stream at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon, wrapped in a numpy array
        :return: the predicted action values (q-values) predicted by the model
        """
        # Generate a one-hot encoded version of the observation if observation space is discrete
        if self._observation_space_type == SpaceType.discrete:
            observation_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[numpy.array(observation_current).reshape(-1)]
        # Return the predicted q-values given the current observation
        return session.run(self._main_network_q_values_predictions,
                           feed_dict={
                               self._main_network_observations: observation_current
                           })

    def get_best_action(self,
                        session,
                        observation_current: numpy.ndarray,
                        possible_actions: [] = None):
        """
        Get the action predicted by the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon, wrapped in a numpy array
        :param possible_actions: the optional list used to bound the actions predicted
        :return: the action predicted by the model
        """
        # Generate a one-hot encoded version of the observation if observation space is discrete
        if self._observation_space_type == SpaceType.discrete:
            observation_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[numpy.array(observation_current).reshape(-1)]
        # Get the predicted action given the current observation
        action = session.run(self._main_network_actions_predicted,
                             feed_dict={
                                   self._main_network_observations: observation_current
                             })
        # If there is a possible action array set the boundaries and scale the action
        if possible_actions is not None:
            lower_bound: numpy.ndarray = possible_actions[:, 0]
            upper_bound: numpy.ndarray = possible_actions[:, 1]
            # Rescale the predicted action in the given boundaries
            for i in range(self._parallel):
                action[i] = (upper_bound[i] - lower_bound[i]) * ((action[i] - (-1)) / (1 - (-1))) + lower_bound[i]
        # Return the rescaled (if necessary) action
        return action

    def get_best_action_and_predicted_action_values(self,
                                                    session,
                                                    observation_current: numpy.ndarray,
                                                    possible_actions: [] = None):
        """
        Get the best action predicted by the model at the given current observation and the predicted action values
        (q-values) according to the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation of the agent in the environment to base prediction upon, wrapped in a numpy array
        :param possible_actions: the optional list used to bound the actions predicted
        :return: the best action predicted by the model and the action values (q-values) predicted by the model
        """
        # Return the best action and all predicted actions values
        return self.get_best_action(session, observation_current, possible_actions), self.get_predicted_action_values(session, observation_current)

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

    def update(self,
               session,
               batch: []):
        """
        Update the model weights (thus training the model) of the policy and q-stream, given a batch of samples.

        :param session: the session of tensorflow currently running
        :param batch: a batch of samples each one consisting in a tuple of observation current, action, reward, observation next and episode done flags
        :return: the q-stream loss and the policy stream loss
        """
        # Unpack the batch into numpy arrays
        observations_current, actions, rewards, observations_next, episode_done_flags = batch[0], batch[1], batch[2], batch[3], batch[4]
        # Generate a one-hot encoded version of the observations if space type is discrete
        if self._observation_space_type == SpaceType.discrete:
            observations_current: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observations_current.reshape(-1)]
            observations_next: numpy.ndarray = numpy.eye(*self._observation_space_shape)[observations_next.reshape(-1)]
        # Get the predicted target q-values over the current observations
        q_values_next = session.run([self._target_network_q_values_predictions],
                                    feed_dict={
                                        self._target_network_observations: observations_next
                                    })[0]
        # Generate the real target values using the Q-Learning Bellman equation
        q_values_target = rewards + self.discount_factor * (1 - episode_done_flags) * q_values_next
        # Q-stream update
        _, q_stream_loss = session.run([self._q_stream_optimizer, self._q_stream_loss],
                                       feed_dict={
                                            self._main_network_observations: observations_current,
                                            self._target_q_values: q_values_target,
                                            self._main_network_actions: actions,
                                       })
        # Policy stream update
        _, policy_stream_loss = session.run([self._policy_stream_optimizer, self._policy_stream_loss],
                                            feed_dict={
                                                    self._main_network_observations: observations_current,
                                            })
        # Return both losses
        return q_stream_loss, policy_stream_loss

    @property
    def warmup_steps(self) -> int:
        return self._buffer_capacity
