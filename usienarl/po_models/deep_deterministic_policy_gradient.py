# Import packages

import tensorflow
import numpy
import scipy.signal

# Import required src

from usienarl import SpaceType, Config, Model
from usienarl.utils.common import softmax


class Buffer:
    """
    A buffer for storing trajectories experienced by a DDPG agent interacting with the environment, it is a simple FIFO
    experience replay buffer.
    """
    def __init__(self,
                 capacity: int,
                 observation_space_shape,
                 action_space_shape):
        # Define buffer attributes
        self._capacity = capacity
        # Define buffer empty attributes
        self._pointer = 0
        self._size = 0
        self._observations_current: numpy.ndarray = numpy.zeros([self._capacity, observation_space_shape], dtype=float)
        self._observations_next: numpy.ndarray = numpy.zeros([self._capacity, observation_space_shape], dtype=float)
        self._actions: numpy.ndarray = numpy.zeros([self._capacity, action_space_shape], dtype=float)
        self._rewards: numpy.ndarray = numpy.zeros(self._capacity, dtype=float)
        self._last_steps: numpy.ndarray = numpy.zeros(self._capacity, dtype=float)

    def store(self, observation_current, action, reward: float, observation_next, last_step: bool):
        """
        Store the time-step in the buffer.

        :param observation_current: the current observation to store in the buffer
        :param action: the last action to store in the buffer
        :param reward: the reward obtained from the action at the current state to store in the buffer
        :param observation_next: the next observation to store in the buffer
        :param last_step: whether or not this time-step was the last of the episode
        """
        # Store data at the index targeted by the pointer
        self._observations_current[self._pointer] = observation_current
        self._observations_next[self._pointer] = observation_next
        self._actions[self._pointer] = action
        self._rewards[self._pointer] = reward
        self._last_steps[self._pointer] = last_step
        # Update the pointer (make sure the first inserted elements are removed first when capacity is exceeded)
        self._pointer = (self._pointer + 1) % self._capacity
        # Update the size with respect to capacity
        self._size = min(self._size + 1, self._capacity)

    def get(self,
            amount: int = 32) -> []:
        """
        Get a batch of data from the buffer of the given size. If size is not given all the buffer is used.

        :param amount: the batch size of data to get
        :return a list containing the ndarrays of: current observations, actions, rewards, next observations and last step flags
        """
        # Adjust the amount with respect to the buffer current size
        if amount <= 0:
            amount = self._size
        # Return a set of random samples of as large as the given amount
        random_indexes: numpy.ndarray = numpy.random.randint(0, self._size, size=amount)
        return [self._observations_current[random_indexes], self._actions[random_indexes], self._rewards[random_indexes], self._observations_next[random_indexes], self._last_steps[random_indexes]]


class Estimator:
    """
    Estimator defining the real Deterministic Policy Gradient (DDPG) model. It is used to define two identical models:
    target network and main-network.

    It is generated given the shape of the observation and action spaces and the hidden layer config defining the
    hidden layers of the network.
    """

    def __init__(self,
                 scope: str,
                 observation_space_shape, agent_action_space_shape,
                 hidden_layers_config: Config):
        self.scope: str = scope
        with tensorflow.variable_scope(self.scope):
            # Define observations placeholder as a float adaptable array with shape Nx(O) where N is the number of examples and (O) the shape of the observation space
            self.observations = tensorflow.placeholder(shape=[None, *observation_space_shape], dtype=tensorflow.float32, name="observations")
            # Define the actions placeholder with adaptable size Nx(A) where N is the number of examples and (A) the shape of the action space
            self.actions = tensorflow.placeholder(shape=[None, *agent_action_space_shape], dtype=tensorflow.float32, name="actions")
            # Define the policy stream
            with tensorflow.variable_scope("policy_stream"):
                # Define the estimator network hidden layers from the config
                hidden_layers_output = hidden_layers_config.apply_hidden_layers(self.observations)
                # Define actions as an array of neurons with same adaptable size Nx(A) and with linear activation functions
                self.actions_predicted = tensorflow.layers.dense(hidden_layers_output, *agent_action_space_shape, name="actions_predicted")
            # Define the q-stream (both for given actions and predicted actions)
            with tensorflow.variable_scope("q_stream"):
                # Define the estimator network q-stream hidden layers from the config
                q_stream_targets_hidden_layers_output = hidden_layers_config.apply_hidden_layers(tensorflow.concat([self.observations, self.actions], axis=-1))
                # Define the estimator network predicted q-values over the predicted actions
                self.q_values_actions = tensorflow.squeeze(q_stream_targets_hidden_layers_output, axis=1, name="q_values_actions")
            with tensorflow.variable_scope("q_stream", reuse=True):
                # Define the estimator network q-stream hidden layers from the config
                q_stream_actions_hidden_layers_output = hidden_layers_config.apply_hidden_layers(tensorflow.concat([self.observations, self.actions_predicted], axis=-1))
                # Define the estimator network predicted q-values over the given actions
                self.q_values_predictions = tensorflow.squeeze(q_stream_actions_hidden_layers_output, axis=1, name="q_values_predictions")
            # Define the estimator weight parameters
            self.weight_parameters = [variable for variable in tensorflow.trainable_variables() if variable.name.startswith(self.scope)]
            self.weight_parameters = sorted(self.weight_parameters, key=lambda parameter: parameter.name)


class DeepDeterministicPolicyGradient(Model):
    """
    Deep Deterministic Policy Gradient Model (DDPG) with FIFO experience replay buffer.-
    The model is a deep neural network which hidden layers can be defined by a config parameter.
    It uses a target network and a main network to correctly evaluate the expected future reward in order
    to stabilize learning.

    In order to synchronize the target network and the main network, every some interval steps the weight have to be
    copied from the main network to the target network.



    Supported observation spaces:
        - continuous

    Supported action spaces:
        - continuous
    """

    def __init__(self,
                 name: str,
                 learning_rate_policy: float, learning_rate_q_values: float,
                 discount_factor: float, polyak_value: float,
                 buffer_capacity: int,
                 hidden_layers_config: Config,
                 error_clipping: bool = True):
        # Define model attributes
        self.learning_rate_policy: float = learning_rate_policy
        self.learning_rate_q_values: float = learning_rate_q_values
        self.discount_factor: float = discount_factor
        self.polyak_value: float = polyak_value
        # Define internal model attributes
        self._buffer_capacity: int = buffer_capacity
        self._hidden_layers_config: Config = hidden_layers_config
        self._error_clipping: bool = error_clipping
        # Define model empty attributes
        self.buffer: Buffer = None
        # Define internal model empty attributes
        self._target_network: Estimator = None
        self._main_network: Estimator = None
        self._main_network_observations = None
        self._main_network_actions = None
        self._main_network_q_values_predictions = None
        self._main_network_q_values_actions = None
        self._target_network_observations = None
        self._target_network_actions = None
        self._target_network_q_values_predictions = None
        self._target_network_q_values_actions = None
        self._rewards = None
        self._episode_done_flags = None
        self._bellman_backup = None
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
        self.buffer = Buffer(self._buffer_capacity, self._observation_space_shape, self._agent_action_space_shape)
        # Define two estimator, one for target network and one for main network, with identical structure
        self._main_network = Estimator(self._scope + "/" + self._name + "/MainNetwork",
                                       self._observation_space_shape, self._agent_action_space_shape,
                                       self._hidden_layers_config)
        self._target_network = Estimator(self._scope + "/" + self._name + "/TargetNetwork",
                                         self._observation_space_shape, self._agent_action_space_shape,
                                         self._hidden_layers_config)
        # Assign main and target networks to the model attributes
        self._main_network_observations = self._main_network.observations
        self._main_network_actions = self._main_network.actions
        self._main_network_q_values_predictions = self._main_network.q_values_predictions
        self._main_network_q_values_actions = self._main_network.q_values_actions
        self._target_network_observations = self._target_network.observations
        self._target_network_actions = self._target_network.actions
        self._target_network_q_values_predictions = self._target_network.q_values_predictions
        self._target_network_q_values_actions = self._target_network.q_values_actions
        # Define the shared part of the model
        with tensorflow.variable_scope(self._scope):
            # Define rewards and done flag placeholders as adaptable arrays with shape N where N is the number of examples
            self.rewards = tensorflow.placeholder(shape=[None, 1], dtype=tensorflow.float32, name="rewards")
            self.episode_done_flags = tensorflow.placeholder(shape=[None, 1], dtype=tensorflow.float32, name="episode_done_flags")
            # Bellman backup for the q-function
            self._bellman_backup = tensorflow.stop_gradient(self._rewards + self.discount_factor * (1 - self._episode_done_flags) * self._target_network_q_values_predictions)
            # Define policy stream loss
            self._policy_stream_loss = -tensorflow.reduce_mean(self._main_network_q_values_predictions, name="policy_stream_loss")
            # Define the absolute error over the q-stream loss
            self._q_stream_absolute_error = tensorflow.abs(self._main_network_q_values_actions - self._bellman_backup, name="q_stream_absolute_error")
            # Define the q-stream loss with error clipping (huber loss) if required
            if self._error_clipping:
                self._q_stream_loss = tensorflow.reduce_mean(tensorflow.where(self._q_stream_absolute_error < 1.0, 0.5 * tensorflow.square(self._q_stream_absolute_error), self._q_stream_absolute_error - 0.5), name="q_stream_loss")
            else:
                self._q_stream_loss = tensorflow.reduce_mean(tensorflow.square(self._q_stream_absolute_error), name="q_stream_loss")
            # Define the optimizer
            self._policy_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_policy).minimize(self._policy_stream_loss)
            self._q_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate_q_values).minimize(self._q_stream_loss)
            # Define the initializer
            self._initializer = tensorflow.global_variables_initializer()
            # Define the weight copy operation (to copy weights from main network to target network at the start)
            self._weight_copier = []
            for main_network_parameter, target_network_parameter in zip(self._main_network.weight_parameters, self._target_network.weight_parameters):
                copy_operation = target_network_parameter.assign(main_network_parameter)
                self._weight_copier.append(copy_operation)
            # Define the weight update operation (to update weight from main network to target network during training using Polyak averaging)
            self._weight_updater = tensorflow.group([tensorflow.assign(target_network_parameter, self.polyak_value * target_network_parameter + (1 - self.polyak_value) * main_network_parameter)
                                                    for main_network_parameter, target_network_parameter in zip(self._main_network.weight_parameters, self._target_network.weight_parameters)])

    def _define_summary(self):
        with tensorflow.variable_scope(self._scope + "/" + self._name):
            self._summary = tensorflow.summary.merge([tensorflow.summary.scalar("policy_stream_loss", self._policy_stream_loss)],
                                                     [tensorflow.summary.scalar("q_stream_loss", self._q_stream_loss)])

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
        # Save by default the observation current input of the model to the given data
        observation_current_input = observation_current
        # Generate a one-hot encoded version of the observation if observation space is discrete
        if self._observation_space_type == SpaceType.discrete:
            observation_current_one_hot: numpy.ndarray = numpy.identity(*self._observation_space_shape)[observation_current]
            observation_current_input = observation_current_one_hot
        # Return all the predicted q-values given the current observation
        return session.run(self._main_network_outputs, feed_dict={self._main_network_observations: [observation_current_input], self._main_network_mask: [mask]})

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

    def copy_weight(self,
                    session):
        """
        Copy the weights from the main network to the target network.

        :param session: the session of tensorflow currently running
        """
        # Run all the weight copy operations
        session.run(self._weight_copier)

    def update_weight(self,
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
        # Unpack the batch into numpy arrays
        observations_current, actions, rewards, observations_next, episode_done_flags, weights = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
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
        # Q-learning update
        _, q_stream_loss = session.run([self._q_stream_optimizer, self._q_stream_loss],
                                       feed_dict={
                                            self._main_network_observations: observations_current_input,
                                            self._target_network_observations: observations_next_input,
                                            self._main_network_actions: actions,
                                            self._rewards: rewards,
                                            self._episode_done_flags: episode_done_flags,
                                       })
        # Policy stream update
        _, policy_stream_loss = session.run([self._policy_stream_optimizer, self._policy_stream_loss],
                                            feed_dict={
                                                    self._main_network_observations: observations_current_input,
                                                    self._target_network_observations: observations_next_input,
                                                    self._main_network_actions: actions,
                                                    self._rewards: rewards,
                                                    self._episode_done_flags: episode_done_flags,
                                            })
        # Return both losses relative summary
        return self._summary, q_stream_loss, policy_stream_loss

    @property
    def warmup_steps(self) -> int:
        return self._buffer_capacity

    @property
    def inputs_name(self) -> str:
        # Get the name of the inputs of the tensorflow graph
        return "MainNetwork/observations"

    @property
    def outputs_name(self) -> str:
        # Get the name of the outputs of the tensorflow graph
        return "MainNetwork/actions_predicted"
