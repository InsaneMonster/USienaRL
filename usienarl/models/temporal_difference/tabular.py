# Import packages

import tensorflow
import numpy

# Import required src

from usienarl import SpaceType, Model


class Buffer:
    pass
    # TODO: BUFFER FOR N-STEPS TD Learning (NO EXPERIENCE REPLAY)


class Tabular(Model):
    """
    Tabular temporal difference model. The weights of the model are the entries of the table and the outputs are computed by
    multiplication of this matrix elements by the inputs.

    It can be updated with Q-Learning update rule or SARSA update rule with respect to the Bellman equation. By default
    it uses Q-Learning.

    Supported observation spaces:
        - discrete

    Supported action spaces:
        - discrete
    """

    def __init__(self,
                 name: str,
                 learning_rate: float, discount_factor: float,
                 use_q_learning: bool = True):
        # Define model attributes
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        # Define internal model attributes
        self._use_q_learning: bool = use_q_learning
        # Define internal tabular model empty attributes
        self._table = None
        # Generate the base model
        super(Tabular, self).__init__(name)
        # Define the types of allowed observation and action spaces
        self._supported_observation_space_types.append(SpaceType.discrete)
        self._supported_action_space_types.append(SpaceType.discrete)

    def _define_graph(self):
        """
        Overridden method of Model class: check its docstring for further information.
        """
        with tensorflow.variable_scope(self.scope + "/" + self.name):
            # Define inputs of the model as a float adaptable array with size Nx(S) where N is the number of examples and (O) is the shape of the observations space
            self._inputs = tensorflow.placeholder(shape=[None, *self.observation_space_shape], dtype=tensorflow.float32, name="inputs")
            # Initialize the table (a weight matrix) of OxA dimensions with random uniform numbers between 0 and 0.1
            self._table = tensorflow.get_variable(name="table", trainable=True, initializer=tensorflow.random_uniform([*self.observation_space_shape, *self.action_space_shape], 0, 0.1))
            # Define the outputs at a given state as a matrix of size NxA given by multiplication of inputs and weights
            # Define the targets for learning with the same Nx(A) adaptable size
            # Note: N is the number of examples and (A) the shape of the action space
            self.outputs = tensorflow.matmul(self._inputs, self._table, name="outputs")
            self._targets = tensorflow.placeholder(shape=[None, *self.action_space_shape], dtype=tensorflow.float32, name="targets")
            # Define the weights of the targets during the update process (e.g. the importance sampling weights)
            self._loss_weights = tensorflow.placeholder(shape=[None, 1], dtype=tensorflow.float32, name="loss_weights")
            # Define the absolute error and its mean
            self._absolute_error = tensorflow.abs(self._targets - self.outputs, name="absolute_error")
            # Define the loss
            self._loss = tensorflow.reduce_sum(self._loss_weights * tensorflow.squared_difference(self._targets, self.outputs), name="loss")
            # Define the optimizer
            self._optimizer = tensorflow.train.GradientDescentOptimizer(self.learning_rate).minimize(self._loss)
            # Define the initializer
            self.initializer = tensorflow.global_variables_initializer()

    def _define_summary(self):
        """
        Overridden method of Model class: check its docstring for further information.
        """
        with tensorflow.variable_scope(self.scope + "/" + self.name):
            # Define the summary operation for this graph with loss and absolute error summaries
            self.summary = tensorflow.summary.merge([tensorflow.summary.scalar("loss", self._loss)])

    def get_output(self,
                   session,
                   observation_current: int):
        """
        Get the outputs of the model at the given current observation.

        :param session: the session of tensorflow currently running
        :param observation_current: the current observation to compute q-values for
        :return: the outputs of the model given the current observation
        """
        # Generate a one-hot encoded version of the observation
        observation_current_one_hot: numpy.ndarray = numpy.identity(*self.observation_space_shape)[observation_current]
        # Return all the predicted q-values given the current observation
        return session.run(self.outputs, feed_dict={self._inputs: [observation_current_one_hot]})

    def predict(self,
                session,
                observation_current: int) -> int:
        """
        Overridden method of Model class: check its docstring for further information.
        """
        # Return the predicted action given the current observation
        return numpy.argmax(self.get_output(session, observation_current))[0]

    def update(self,
               session,
               current_episode: int, total_episodes: int, current_step: int,
               batch: [], sample_weights: numpy.ndarray):
        """
        Overridden method of Model class: check its docstring for further information.
        """
        # Unpack the batch into numpy arrays
        observations_current, actions, rewards, observations_next = batch[0], batch[1], batch[2], batch[3]
        # Generate a one-hot encoded version of the observations
        observations_current_one_hot: numpy.ndarray = numpy.eye(*self.observation_space_shape)[observations_current.reshape(-1)]
        observations_next_one_hot: numpy.ndarray = numpy.eye(*self.observation_space_shape)[observations_next.reshape(-1)]
        # Get the q-values from the model at both current observations and next observations
        q_values_current: numpy.ndarray = session.run(self.outputs, feed_dict={self._inputs: observations_current_one_hot})
        q_values_next: numpy.ndarray = session.run(self.outputs, feed_dict={self._inputs: observations_next_one_hot})
        # Apply Bellman equation with the required update rule (Q-Learning or SARSA)
        if self._use_q_learning:
            self._q_learning_update_rule(len(batch), observations_next, actions, q_values_current, q_values_next, rewards, self.discount_factor)
        else:
            self._sarsa_update_rule(len(batch), observations_next, actions, q_values_current, q_values_next, rewards, self.discount_factor)
        # Train the model and save the value of the loss and of the absolute error as well as the summary
        _, loss, absolute_error, summary = session.run([self._optimizer, self._loss, self._absolute_error, self.summary],
                                                       feed_dict={
                                                                   self._inputs: observations_current_one_hot,
                                                                   self._targets: q_values_current,
                                                                   self._loss_weights: sample_weights
                                                                  })
        # Return the loss, the absolute error and relative summary
        return summary, loss, absolute_error

    @staticmethod
    def get_inputs_name() -> str:
        """
        Overridden method of Model class: check its docstring for further information.
        """
        # Get the name of the inputs of the tensorflow graph
        return "inputs"

    @staticmethod
    def get_outputs_name() -> str:
        """
        Overridden method of Model class: check its docstring for further information.
        """
        # Get the name of the outputs of the tensorflow graph
        return "outputs"

    @staticmethod
    def _q_learning_update_rule(batch_size: int,
                                observations_next: numpy.ndarray, actions: numpy.ndarray,
                                q_values_current: numpy.ndarray, q_values_next: numpy.ndarray,
                                rewards: numpy.ndarray, discount_factor: float):
        """
        Update the Q-Values target to be estimated by the model using q-learning update rule for the Bellman equation:
        Q(s, a) = R + gamma * max_a(Q(s'))

        :param observations_next: the next observation for each sample
        :param actions: the actions taken by the agent at each sample
        :param q_values_current: the q-values output for the current observation (the target to be updated)
        :param q_values_next: the q-values output for the next observation (used to update the target)
        :param rewards: the rewards obtained by the agent at each sample
        :param discount_factor: the discount factor set for the model, i.e. gamma
        """
        for sample_index in range(batch_size):
            # Extract current sample values
            observation_next: int = observations_next[sample_index]
            action: int = actions[sample_index]
            reward: float = rewards[sample_index]
            # Update the q-values for current observation using Q-Learning Bellman equation
            # Note: only the immediate reward can be assigned at end of the episode, i.e. when next observation is None
            if observation_next is None:
                q_values_current[sample_index, action] = reward
            else:
                q_values_current[sample_index, action] = rewards + discount_factor * numpy.max(q_values_next[sample_index])

    @staticmethod
    def _sarsa_update_rule(batch_size: int,
                           observations_next: numpy.ndarray, actions: numpy.ndarray,
                           q_values_current: numpy.ndarray, q_values_next: numpy.ndarray,
                           rewards: numpy.ndarray, discount_factor: float):
        """
        Update the Q-Values target to be estimated by the model using SARSA update rule for the Bellman equation:
        Q(s, a) = R + gamma * Q(s', a)

        :param observations_next: the next observation for each sample
        :param actions: the actions taken by the agent at each sample
        :param q_values_current: the q-values output for the current observation (the target to be updated)
        :param q_values_next: the q-values output for the next observation (used to update the target)
        :param rewards: the rewards obtained by the agent at each sample
        :param discount_factor: the discount factor set for the model, i.e. gamma
        """
        for sample_index in range(batch_size):
            # Extract current sample values
            observation_next: int = observations_next[sample_index]
            action: int = actions[sample_index]
            reward: float = rewards[sample_index]
            # Update the q-values for current observation using SARSA Bellman equation
            # Note: only the immediate reward can be assigned at end of the episode, i.e. when next observation is None
            if observation_next is None:
                q_values_current[sample_index, action] = reward
            else:
                q_values_current[sample_index, action] = rewards + discount_factor * q_values_next[sample_index, action]
