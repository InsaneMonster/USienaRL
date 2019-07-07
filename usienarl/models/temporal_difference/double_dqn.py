# Import packages

import tensorflow
import numpy

# Import required src

from usienarl import SpaceType, Config
from usienarl.models import TemporalDifferenceModel


class Estimator:
    """
    Estimator defining the real DDQN model. It is used to define two identical models: target network and q-network.

    It is generated given the size of the observation and action spaces and the hidden layer config defining the
    hidden layers of the DDQN.
    """

    def __init__(self,
                 scope: str,
                 observation_space_shape, action_space_shape,
                 hidden_layers_config: Config):
        self.scope = scope
        with tensorflow.variable_scope(self.scope):
            # Define inputs of the estimator as a float adaptable array with shape Nx(S) where N is the number of examples and (S) the shape of the state
            self.inputs = tensorflow.placeholder(shape=[None, *observation_space_shape], dtype=tensorflow.float32, name="inputs")
            # Define the estimator network hidden layers from the config
            hidden_layers_output = hidden_layers_config.apply_hidden_layers(self.inputs)
            # Define outputs as an array of neurons of size NxA and with linear activation functions
            # Define the targets for learning with the same NxA adaptable size
            # Note: N is the number of examples and A the size of the action space (DDQN only supports discrete actions spaces)
            self.outputs = tensorflow.layers.dense(hidden_layers_output, *action_space_shape, name="outputs")
            self.targets = tensorflow.placeholder(shape=[None, *action_space_shape], dtype=tensorflow.float32, name="targets")
            # Define the weights of the targets during the update process (e.g. the importance sampling weights)
            self.loss_weights = tensorflow.placeholder(shape=[None, 1], dtype=tensorflow.float32, name="loss_weights")
            # Define the absolute error
            self.absolute_error = tensorflow.abs(self.targets - self.outputs, name="absolute_error")
            # Define the estimator loss
            self.loss = tensorflow.reduce_mean(self.loss_weights * tensorflow.squared_difference(self.targets, self.outputs), name="loss")
            # Define the estimator weight parameters
            self.weight_parameters = [variable for variable
                                      in tensorflow.trainable_variables()
                                      if variable.name.startswith(self.scope)]
            self.weight_parameters = sorted(self.weight_parameters, key=lambda parameter: parameter.name)


class DoubleDQN(TemporalDifferenceModel):
    """
    DDQN (Double Deep Q-Network) model. The model is a deep neural network which hidden layers can be defined by a config
    parameter. It uses a target network and a q-network to correctly evaluate the expected future reward in order
    to stabilize learning.

    In order to synchronize the target network and the q-network, every some interval steps the weight have to be copied
    from the q-network to the target network.

    It is further enhanced by the update process of the q-value. In the DDQN the q-network estimates the outputs given
    the current state, but the best predicted action (the index of the best predicted output) is chosen by the target
    network.

    Supported observation spaces:
        - discrete
        - continuous

    Supported action spaces:
        - discrete
    """

    def __init__(self,
                 name: str,
                 learning_rate: float, discount_factor: float,
                 hidden_layers_config: Config,
                 weight_copy_step_interval: int):
        # Define Double DQN attributes
        self._target_network: Estimator = None
        self._q_network: Estimator = None
        self._outputs_target = None
        self._inputs_target = None
        self._hidden_layers_config: Config = hidden_layers_config
        self._weight_copy_step_interval: int = weight_copy_step_interval
        # Generate the base Q-Learning model
        super().__init__(name, learning_rate, discount_factor)
        # Define the types of allowed observation and action spaces
        self._supported_observation_space_types.append(SpaceType.discrete)
        self._supported_observation_space_types.append(SpaceType.continuous)
        self._supported_action_space_types.append(SpaceType.discrete)

    def _define_graph(self):
        """
        Overridden method of Model class: check its docstring for further information.
        """
        # Define two estimator, one for target network and one for q-network, with identical structure
        self._q_network = Estimator(self.scope + "/" + self.name + "/QNetwork",
                                    self.observation_space_shape, self.action_space_shape,
                                    self._hidden_layers_config)
        self._target_network = Estimator(self.scope + "/" + self.name + "/TargetNetwork",
                                         self.observation_space_shape, self.action_space_shape,
                                         self._hidden_layers_config)
        # Assign the q-network properties to the model properties (q-network is the actual model)
        self._inputs = self._q_network.inputs
        self.outputs = self._q_network.outputs
        self._targets = self._q_network.targets
        self._absolute_error = self._q_network.absolute_error
        self._loss = self._q_network.loss
        self._loss_weights = self._q_network.loss_weights
        # Assign the target network outputs and inputs to the specific target outputs/inputs of the model
        self._outputs_target = self._target_network.outputs
        self._inputs_target = self._target_network.inputs
        # Define the optimizer
        self._optimizer = tensorflow.train.AdamOptimizer(self.learning_rate).minimize(self._loss)
        # Define the initializer
        self.initializer = tensorflow.global_variables_initializer()
        # Define the weight copy operation (to copy weights from q-network to target network)
        self._weight_copier = []
        for q_network_parameter, target_network_parameter in zip(self._q_network.weight_parameters, self._target_network.weight_parameters):
            copy_operation = target_network_parameter.assign(q_network_parameter)
            self._weight_copier.append(copy_operation)

    def _copy_weight(self,
                     session,
                     step: int = -1):
        """
        Copy the weights from the q-network to the target network depending on the current step.

        :param session: the session of tensorflow currently running
        :param step: the current step in the training process.
        """
        # Check if should update the target estimator at this step
        if step == -1 or step % self._weight_copy_step_interval == 0:
            # Run all the weight copy operations
            session.run(self._weight_copier)

    def train_initialize(self,
                         session: object):
        """
        Overridden method of QLearningModel class: check its docstring for further information.
        """
        # Copy weights of q-network to target network to make them the same at the beginning
        self._copy_weight(session)

    def update_single(self,
                      session,
                      episode: int, episodes: int, step: int,
                      state_current, state_next, action: int, reward: float, sample_weight: float = 1.0):
        """
        Overridden method of QLearningModel class: check its docstring for further information.
        """
        # Copy the weight of the q-network in the target network, if required at this episode/step
        self._copy_weight(session, step)
        # Get the outputs depending on the type of space (discrete is one-hot encoded)
        if self.observation_space_type == SpaceType.discrete:
            # Get the outputs at the current state and at the next state, with next state computed by the target network
            # Note: next states are computed by both the q-network and the target network
            outputs_current = session.run(self.outputs,
                                          feed_dict={
                                              self._inputs: [numpy.identity(*self.observation_space_shape)[state_current]]})
            outputs_next_q_network = session.run(self.outputs,
                                                 feed_dict={self._inputs: [numpy.identity(*self.observation_space_shape)
                                                            [state_next if state_next is not None else 0]]})
            outputs_next_target_network = session.run(self._outputs_target,
                                                      feed_dict={self._inputs_target: [numpy.identity(*self.observation_space_shape)
                                                                 [state_next if state_next is not None else 0]]})
        else:
            # Get the outputs at the current state and at the next state, with next state computed by the target network
            # Note: next states are computed by both the q-network and the target network
            outputs_current = session.run(self.outputs,
                                          feed_dict={self._inputs: [state_current]})
            outputs_next_q_network = session.run(self.outputs,
                                                 feed_dict={self._inputs: [state_next if state_next is not None
                                                                           else numpy.zeros(
                                                                                self.observation_space_shape,
                                                                                dtype=float)]})
            outputs_next_target_network = session.run(self._outputs_target,
                                                      feed_dict={self._inputs_target: [state_next if state_next is not None
                                                                                       else numpy.zeros(
                                                                                            self.observation_space_shape,
                                                                                            dtype=float)]})
        # Apply Bellman equation, modifying the weights at the current state with the discounted future reward of the
        # next state given the action
        if state_next is None:
            # Only the immediate reward can be assigned at end of the episode
            outputs_current[0, action] = reward
        else:
            # Predict the output using the q-network and then get the q-value estimated by the target network at the
            # same index
            predicted_output_index: int = numpy.argmax(outputs_next_q_network)
            outputs_current[0, action] = reward + self.discount_factor * outputs_next_target_network[predicted_output_index]
        # Run the optimizer while also evaluating new weights values
        # Note: the current outputs modified by the Bellman equation are now used as target for the model
        # Save the value of the loss and of the absolute error as well as the summary
        # Input values changes according to observation space type (discrete is one-hot encoded)
        if self.observation_space_type == SpaceType.discrete:
            _, loss, absolute_error, summary = session.run([self._optimizer, self._loss, self._absolute_error, self.summary],
                                                           feed_dict={
                                                           self._inputs: [numpy.identity(self.observation_space_shape)[state_current]],
                                                           self._targets: outputs_current,
                                                           self._loss_weights: [[sample_weight]]
                                                           })
        else:
            _, loss, absolute_error, summary = session.run([self._optimizer, self._loss, self._absolute_error, self.summary],
                                                           feed_dict={
                                                           self._targets: outputs_current,
                                                           self._loss_weights: [[sample_weight]]
                                                           })
        # Return the loss, the absolute error and the summary
        return loss, absolute_error, summary

    def update_batch(self,
                     session,
                     episode: int, episodes: int, step: int,
                     batch: [], sample_weights: []):
        """
        Overridden method of QLearningModel class: check its docstring for further information.
        """
        # Copy the weight of the q-network in the target network, if required at this episode/step
        self._copy_weight(session, step)
        # Get the outputs depending on the type of space (discrete is one-hot encoded)
        if self.observation_space_type == SpaceType.discrete:
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
                                       else numpy.zeros(self.observation_space_shape, dtype=float) for val in
                                       batch])
        # Get the outputs at the current states and at the next states
        # Note: next states are computed by both the q-network and the target network
        outputs_current = session.run(self.outputs,
                                      feed_dict={self._inputs: states_current})
        outputs_next_q_network = session.run(self.outputs,
                                             feed_dict={self._inputs: states_next})
        outputs_next_target_network = session.run(self._outputs_target,
                                                  feed_dict={self._inputs_target: states_next})
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
                # Predict the output using the q-network and then get the q-value estimated by the target network at the
                # same index
                predicted_output_index: int = numpy.argmax(outputs_next_q_network[i])
                outputs_current[i, action] = reward + self.discount_factor * outputs_next_target_network[i][predicted_output_index]
            # Insert training data in training arrays depending on the observation space type
            if self.observation_space_type == SpaceType.discrete:
                inputs[i] = numpy.identity(*self.observation_space_shape)[state_current]
            else:
                inputs[i] = state_current
            # The current outputs modified by the Bellman equation are now used as target for the model
            targets[i] = outputs_current[i]
        # Feed the training arrays into the network and run the optimizer while also evaluating new weights values
        # Save the value of the loss and of the absolute error as well as the summary
        _, loss, absolute_error, summary = session.run([self._optimizer, self._loss, self._absolute_error, self.summary],
                                                       feed_dict={
                                                       self._inputs: inputs,
                                                       self._targets: targets,
                                                       self._loss_weights: sample_weights
                                                       })
        # Return the loss, the absolute error and the summary
        return loss, absolute_error, summary

    @staticmethod
    def get_inputs_name() -> str:
        """
        Overridden method of QLearningModel class: check its docstring for further information.
        """
        # Get the name of the inputs of the tensorflow graph
        return "QNetwork/inputs"

    @staticmethod
    def get_outputs_name() -> str:
        """
        Overridden method of QLearningModel class: check its docstring for further information.
        """
        # Get the name of the outputs of the tensorflow graph
        return "QNetwork/outputs"
