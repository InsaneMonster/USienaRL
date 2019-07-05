# Import packages

import tensorflow
import numpy

# Import required src

from usienarl import SpaceType
from usienarl.models import TemporalDifferenceModel


class QTable(TemporalDifferenceModel):
    """
    Q-Table model. The weights of the model are the entries of the so called Q-Table and the outputs are computed by
    multiplication of this matrix elements by the inputs.

    Supported observation spaces:
        - discrete

    Supported action spaces:
        - discrete
    """

    def __init__(self,
                 name: str,
                 learning_rate: float, discount_factor: float):
        # Define Q-Table attributes
        self._q_table = None
        # Generate the base Temporal Difference model
        super().__init__(name, learning_rate, discount_factor)
        # Define the types of allowed observation and action spaces
        self._supported_observation_space_types.append(SpaceType.discrete)
        self._supported_action_space_types.append(SpaceType.discrete)

    def _define_graph(self):
        """
        Overridden method of Model class: check its docstring for further information.
        """
        with tensorflow.variable_scope(self._experiment_name + "/" + self.name):
            # Define inputs of the model as a float adaptable array with size Nx(S) where N is the number of examples and (S) is the shape of the observations space
            self._inputs = tensorflow.placeholder(shape=[None, *self.observation_space_shape], dtype=tensorflow.float32, name="inputs")
            # Initialize the Q-Table (a weight matrix) of SxA dimensions with random uniform numbers between 0 and 0.1
            self._q_table = tensorflow.get_variable(
                                name="q_table",
                                trainable=True,
                                initializer=tensorflow.random_uniform([*self.observation_space_shape, *self.action_space_shape], 0, 0.1))
            # Define the outputs at a given state as a matrix of size NxA given by multiplication of inputs and weights
            # Define the targets for learning with the same Nx(A) adaptable size
            # Note: N is the number of examples and (A) the shape of the action space
            self.outputs = tensorflow.matmul(self._inputs, self._q_table, name="outputs")
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

    def update_single(self,
                      session,
                      episode: int, episodes: int, step: int,
                      state_current, state_next, action: int, reward: float, sample_weight: float = 1.0):
        """
        Overridden method of TemporalDifferenceModel class: check its docstring for further information.
        """
        # Get the outputs depending on the type of space (discrete is one-hot encoded)
        if self._observation_space_type == SpaceType.discrete:
            # Get the outputs at the current state and at the next state
            outputs_current = session.run(self.outputs,
                                          feed_dict={self._inputs: [numpy.identity(*self.observation_space_shape)[state_current]]})
            outputs_next = session.run(self.outputs,
                                       feed_dict={self._inputs: [numpy.identity(*self.observation_space_shape)
                                                                 [state_next if state_next is not None else 0]]})
        else:
            # Get the outputs at the current state and at the next state
            outputs_current = session.run(self.outputs,
                                          feed_dict={self._inputs: [state_current]})
            outputs_next = session.run(self.outputs,
                                       feed_dict={self._inputs: [state_next if state_next is not None
                                                                 else numpy.zeros(
                                                                    self.observation_space_shape,
                                                                    dtype=float)]})
        # Apply Bellman equation, modifying the weights at the current state with the discounted future reward of the
        # next state given the action
        if state_next is None:
            # Only the immediate reward can be assigned at end of the episode
            outputs_current[0, action] = reward
        else:
            outputs_current[0, action] = reward + self.discount_factor * numpy.max(outputs_next)
        # Run the optimizer while also evaluating new weights values
        # Note: the current outputs modified by the Bellman equation are now used as target for the model
        # Save the value of the loss and of the absolute error as well as the summaries
        # Input values changes according to observation space type (discrete is one-hot encoded)
        if self._observation_space_type == SpaceType.discrete:
            _, loss, absolute_error, summary = session.run([self._optimizer, self._loss, self._absolute_error, self.summary],
                                                           feed_dict={
                                                           self._inputs: [numpy.identity(*self.observation_space_shape)[state_current]],
                                                           self._targets: outputs_current,
                                                           self._loss_weights: [[sample_weight]]
                                                           })
        else:
            _, loss, absolute_error, summary = session.run([self._optimizer, self._loss, self._absolute_error, self.summary],
                                                           feed_dict={
                                                           self._inputs: [state_current],
                                                           self._targets: outputs_current,
                                                           self._loss_weights: [[sample_weight]]
                                                           })
        # Return the loss, the absolute error and relative summaries
        return loss, absolute_error, summary

    def update_batch(self,
                     session,
                     episode: int, episodes: int, step: int,
                     batch: [], sample_weights: []):
        """
        Overridden method of TemporalDifferenceModel class: check its docstring for further information.
        """
        # Get the outputs depending on the type of space (discrete is one-hot encoded)
        if self._observation_space_type == SpaceType.discrete:
            # Get all current states in the batch
            states_current = numpy.array([numpy.identity(*self.observation_space_shape)[val[0]] for val in batch])
            # Get all next states in the batch (if equals to None, it means end of episode, and as such no next state)
            states_next = numpy.array([numpy.identity(*self.observation_space_shape)[val[3] if val[3] is not None else 0]
                                      for val in batch])
        else:
            # Get all current states in the batch
            states_current = numpy.array([val[0] for val in batch])
            # Get all next states in the batch (if equals to None, it means end of episode, and as such no next state)
            states_next = numpy.array([val[3] if val[3] is not None
                                       else numpy.zeros(self.observation_space_shape, dtype=float) for val in batch])
        # Get the outputs at the current states and at the next states
        outputs_current = session.run(self.outputs,
                                      feed_dict={self._inputs: states_current})
        outputs_next = session.run(self.outputs,
                                   feed_dict={self._inputs: states_next})
        # Define training arrays
        inputs = numpy.zeros((len(batch), *self.observation_space_shape))
        targets = numpy.zeros((len(batch), *self.action_space_shape))
        for i, example in enumerate(batch):
            state_current, action, reward, state_next = example[0], example[1], example[2], example[3]
            # Apply Bellman equation, modifying the weights at the current states with the discounted future reward of
            # the next states given the actions
            if state_next is None:
                # Only the immediate reward can be assigned at end of the episode
                outputs_current[i, action] = reward
            else:
                outputs_current[i, action] = reward + self.discount_factor * numpy.max(outputs_next[i])
            # Insert training data in training arrays depending on the observation space type
            if self._observation_space_type == SpaceType.discrete:
                inputs[i] = numpy.identity(*self.observation_space_shape)[state_current]
            else:
                inputs[i] = state_current
            # The current outputs modified by the Bellman equation are now used as target for the model
            targets[i] = outputs_current[i]
        # Feed the training arrays into the network and run the optimizer while also evaluating new weights values
        # Save the value of the loss and of the absolute error as well as the summaries
        _, loss, absolute_error, summary = session.run([self._optimizer, self._loss, self._absolute_error, self.summary],
                                                       feed_dict={
                                                       self._inputs: inputs,
                                                       self._targets: targets,
                                                       self._loss_weights: sample_weights
                                                       })
        # Return the loss, the absolute error and relative summary
        return loss, absolute_error, summary

    @staticmethod
    def get_inputs_name() -> str:
        """
        Overridden method of TemporalDifferenceModel class: check its docstring for further information.
        """
        # Get the name of the inputs of the tensorflow graph
        return "inputs"

    @staticmethod
    def get_outputs_name() -> str:
        """
        Overridden method of TemporalDifferenceModel class: check its docstring for further information.
        """
        # Get the name of the outputs of the tensorflow graph
        return "outputs"

    @staticmethod
    def _q_learning_update_rule(state_next, action, q_values_current, q_values_next, reward: float, discount_factor: float):
        """
        Update the Q-Values target to be estimated by the model using q-learning update rule for the Bellman equation:
        Q(s, a) = R + gamma * max_a(Q(s'))

        :param state_next: the next state observed by the agent
        :param action: the action taken by the agent
        :param q_values_current: the q-values output of the approximator for the current state (the target to be updated)
        :param q_values_next: the q-values output of the approximator for the next state
        :param reward: the reward got by the agent
        :param discount_factor: the discount factor set for the model (in literature known as gamma)
        """
        # Update the q-values for current state (the model target) using Q-Learning Bellman equation
        if state_next is None:
            # Only the immediate reward can be assigned at end of the episode
            q_values_current[action] = reward
        else:
            q_values_current[action] = reward + discount_factor * numpy.max(q_values_next)

    @staticmethod
    def _sarsa_update_rule(state_next, action, q_values_current, q_values_next, reward: float, discount_factor: float):
        """
        Update the Q-Values target to be estimated by the model using SARSA update rule for the Bellman equation:
        Q(s, a) = R + gamma * Q(s', a)

        :param state_next: the next state observed by the agent
        :param action: the action taken by the agent
        :param q_values_current: the q-values output of the approximator for the current state (the target to be updated)
        :param q_values_next: the q-values output of the approximator for the next state
        :param reward: the reward got by the agent
        :param discount_factor: the discount factor set for the model (in literature known as gamma)
        """
        # Update the q-values for current state (the model target) using SARSA Bellman equation
        if state_next is None:
            # Only the immediate reward can be assigned at end of the episode
            q_values_current[action] = reward
        else:
            q_values_current[action] = reward + discount_factor * q_values_next[action]
