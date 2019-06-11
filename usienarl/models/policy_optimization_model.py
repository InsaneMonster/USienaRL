# Import packages

import numpy
import tensorflow

# Import required src

from usienarl import Model


class PolicyOptimizationModel(Model):
    """
    Base class for each Policy Optimization model. It should not be used by itself, and should be extended instead.

    A Policy Optimization model is a model computing directly the gradient of the policy by samples drawn from the
    execution of a certain amount of episodes (a batch) using the last found optimal policy.
    It usually can use both discrete and continuous action spaces.

    Policy optimization models always predict at least both the action and estimate the value on the given state.
    Because of that, usually policy optimization models uses two stream or sub-models for that.

    Attributes:
        - discount_factor: the discount factor of the discounted future expected reward
        - buffer: the buffer in which to store the experiences to train the network
    """

    def __init__(self,
                 name: str,
                 discount_factor: float,):
        # Set discount factor
        self.discount_factor: float = discount_factor
        # Define empty Policy Optimization models attributes
        self._inputs = None
        self._actions = None
        self._targets = None
        self._advantages = None
        self._rewards = None
        self._log_likelihood = None
        self._value = None
        self._value_stream_loss = None
        self._policy_stream_loss = None
        self._value_stream_optimizer = None
        self._policy_stream_optimizer = None
        # Define buffer
        self.buffer = None
        # Generate the base model
        super().__init__(name)

    def update(self,
               session,
               batch: []):
        """
        Update the model with data contained in the given batch list.

        :param session: the session of tensorflow currently running
        :param batch: the batch list consisting of data used for training (usually a sequence of trajectories)
        :return: losses of the model and summary
        """
        # Empty method, definition should be implemented on a child class basis
        return None

    def get_trainable_variables(self,
                                scope: str):
        """
        Get the trainable variables in the model (useful for saving the model or comparing the weights), given the
        current experiment scope.

        :param scope: the string scope of the tensorflow graph (usually the name of the experiment)
        :return: the trainable tensorflow variables.
        """
        # Get the training variables of the model under its scope: usually, the training variables of the tensorflow graph
        return tensorflow.trainable_variables(scope + "/" + self.name + "/")

    def print_trainable_variables(self,
                                  scope: str,
                                  session):
        """
        Print the trainable variables in the current tensorflow graph, given the current experiment scope.

        :param scope: the string scope of the tensorflow graph (usually the name of the experiment)
        :param session: the session of tensorflow currently running
        """
        # Print the trainable variables of the currently active model
        trainable_variables = self.get_trainable_variables(scope)
        for trainable_variable in trainable_variables:
            print(trainable_variable.name)
            print(trainable_variable.eval(session=session))

    @staticmethod
    def get_categorical_log_likelihood(target_actions_mask, logits):
        """
        Get log-likelihood for discrete action spaces (using a categorical distribution) on the given logits and with
        the target action mask.
        It uses tensorflow and as such should only be called in the _define method of the child class.

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
        It uses tensorflow and as such should only be called in the _define method of the child class.

        :param target_actions: the target actions used to compute the log-likelihood tensor
        :param expected_value: the expected value of the gaussian distribution
        :param log_std: the log-std of the gaussian distribution
        :return: the log-likelihood according to gaussian distribution
        """
        # Define the log-likelihood tensor for the gaussian distribution on the given target actions
        log_likelihood_tensor = -0.5 * (((target_actions - expected_value) / (tensorflow.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + numpy.log(2 * numpy.pi))
        # Return the gaussian log-likelihood by summing over all the elements in the log-likelihood tensor defined above
        return tensorflow.reduce_sum(log_likelihood_tensor, axis=1, name="log_likelihood")

    @staticmethod
    def get_diagonal_gaussian_kl(expected_value_a, log_std_a, expected_value_b, log_std_b):
        """
        Compute mean KL divergence between two batches of diagonal gaussian distributions,
        where distributions are specified by means and log-std.
        """
        # Define to variable as the exponential of the two log-std multiplied by two
        var_a, var_b = tensorflow.exp(2 * log_std_a), tensorflow.exp(2 * log_std_b)
        # Define KL divergences between the two distributions
        kl_divergences = tensorflow.reduce_sum(0.5 * (((expected_value_b - expected_value_a) ** 2 + var_a) / (var_b + 1e-8) - 1) + log_std_b - log_std_a, axis=1)
        # Return the mean of the KL divergences
        return tensorflow.reduce_mean(kl_divergences)

    @staticmethod
    def get_categorical_kl(log_likelihood_a, log_likelihood_b):
        """
        Compute mean KL divergence between two batches of categorical probability distributions,
        where the distributions are input as log-likelihoods.

        :param log_likelihood_a: the first log-likelihood operand of the KL divergence
        :param log_likelihood_b: the second log-likelihood operand of the KL divergence
        :return the mean KL divergence on the operands
        """
        # Define KL divergences between the two log-likelihood distributions
        kl_divergences = tensorflow.reduce_sum(tensorflow.exp(log_likelihood_b) * (log_likelihood_b - log_likelihood_a), axis=1)
        # Return the mean of the KL divergences
        return tensorflow.reduce_mean(kl_divergences)
