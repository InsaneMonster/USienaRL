# Import packages

import numpy
import tensorflow

# Import required src

from usienarl import Model
from usienarl import SpaceType


class TemporalDifferenceModel(Model):
    """
    Base class for each Temporal Difference (TD) model. It should not be used by itself, and should be extended instead.

    A TD model is a model which uses the Q-value to find the optimal policy at each step by temporal difference.
    It usually supports only discrete action spaces, but it depends on the model.

    Attributes:
        - learning_rate: the lambda of the model in training phase
        - discount_factor: the discount factor of the discounted future expected reward
        - outputs: the outputs of the model (usually one-hot encoded in the size of the action space of the environment)
    """

    def __init__(self,
                 name: str,
                 learning_rate: float, discount_factor: float):
        # Set learning rate and discount factor
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        # Define empty Q-Learning models attributes
        self._inputs = None
        self._absolute_error = None
        self._loss = None
        self._loss_weights = None
        self._optimizer = None
        self.outputs = None
        self._targets = None
        # Generate the base model
        super().__init__(name)

    def _define_summary(self):
        """
        Overridden method of Model class: check its docstring for further information.
        """
        with tensorflow.variable_scope(self._experiment_name + "/" + self.name):
            # Define the summary operation for this graph with loss and absolute error summaries
            self.summary = tensorflow.summary.merge([tensorflow.summary.scalar("loss", self._loss)])

    def train_initialize(self,
                         session: object):
        """
        Initialize the training (e.g. to setup the weights of the model).

        :param session: the session of tensorflow currently running.
        """
        # Empty method, definition should be implemented on a child class basis
        pass

    def update_single(self,
                      session,
                      episode: int, episodes: int, step: int,
                      state_current, state_next, action: int, reward: float, sample_weight: float = 1.0):
        """
        Update the model weights (thus training the model) of only one step experience in the environment.

        :param session: the session of tensorflow currently running.
        :param episode: the current episode number in the experiment
        :param episodes: the total episodes number in the experiment
        :param step: the current step number in the experiment
        :param state_current: the current state observed in the environment
        :param state_next: the next state reached in the environment
        :param action: the action take to transition between the state_current and the next_state
        :param reward: the reward obtained for taking the action at the state_current
        :param sample_weight: the weight of this sample
        :return: the loss of the model after training, the absolute error after training and the updated summary
        """
        # Empty method, definition should be implemented on a child class basis
        return None

    def update_batch(self,
                     session,
                     episode: int, episodes: int, step: int,
                     batch: [], sample_weights: []):
        """
        Update the model weights (thus training the model) of a batch of step experiences in the environment.

        :param session: the session of tensorflow currently running
        :param episode: the current episode number in the experiment
        :param episodes: the total episodes number in the experiment
        :param step: the current step number in the experiment
        :param batch: a batch of samples each one consisting of a tuple like: (state_current, action, reward, state_next)
        :param sample_weights: the weights of each sample in the batch
        :return: the loss of the model after training, the absolute error after training and the updated summary
        """
        # Empty method, definition should be implemented on a child class basis
        return None

    def get_output(self,
                   session,
                   state_current):
        """
        Get the outputs of the model at the given current state.

        :param session: the session of tensorflow currently running
        :param state_current: the current state in the environment to get the output for
        :return: the outputs of the model given the state_current
        """
        # Return all the predicted actions q-values given the current state depending on the observation space type
        if self._observation_space_type == SpaceType.discrete:
            return session.run(self.outputs,
                               feed_dict={self._inputs: [numpy.identity(*self.observation_space_shape)[state_current]]})
        else:
            return session.run(self.outputs,
                               feed_dict={self._inputs: [state_current]})

    def predict(self,
                session,
                state_current) -> []:
        """
        Overridden method of Model class: check its docstring for further information.
        """
        # Return the best predicted action (max output index) given the current state in the shape of a list
        return [numpy.argmax(self.get_output(session, state_current))]

    @staticmethod
    def get_inputs_name() -> str:
        """
        Get the inputs name in the model. It is the same for each model of the same model type.

        :return: the str defining the inputs name for the model
        """
        # Empty method, definition should be implemented on a child class basis
        return ''

    @staticmethod
    def get_outputs_name() -> str:
        """
        Get the outputs name in the model. It is the same for each model of the same model type.

        :return: the str defining the outputs name for the model
        """
        # Empty method, definition should be implemented on a child class basis
        return ''





