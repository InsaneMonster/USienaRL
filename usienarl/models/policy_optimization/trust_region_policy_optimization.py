# Import packages

import tensorflow
import numpy
import scipy.signal

# Import required src

from usienarl import SpaceType, Config
from usienarl.models import PolicyOptimizationModel


class Buffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting with the environment,
    and using Generalized Advantage Estimation (GAE-Lambda) for calculating the advantages of state-action pairs.

    The buffer is dynamically resizable.

    The buffer contains list of states (or observations), actions (used as targets), values (computed by the value stream
    of the model itself during prediction), advantages (computed by the buffer using GAE when a trajectory finishes and
    fed back up in the policy stream to drive the loss), rewards (used to compute rewards-to-go) and rewards-to-go
    (computed inside the buffer itself and used as weight for the targets action when training the value stream).
    """

    def __init__(self,
                 action_space_type: SpaceType,
                 discount_factor: float, lambda_parameter: float):
        # Define action space type
        self._action_space_type: SpaceType = action_space_type
        # Define buffer components for all action space type
        self._states: [] = []
        self._actions: [] = []
        self._advantages: [] = []
        self._rewards: [] = []
        self._rewards_to_go: [] = []
        self._values: [] = []
        self._log_likelihoods_unmasked: [] = []
        self._expected_values: [] = []
        self._log_stds: [] = []
        # Define parameters
        self._discount_factor: float = discount_factor
        self._lambda_parameter: float = lambda_parameter
        # Define buffer pointer
        self._pointer: int = 0
        self._path_start_index: int = 0

    def store(self,
              state, action, reward: float, value: float, *args):
        """
        Store the time-step in the buffer.
        Note: the args optional parameter changes depending on the space type.

        Discrete space type:
            - log-likelihood unmasked as first and only argument

        Continuous space type:
            - expected value as first argument
            - log-std as second argument

        :param state: the current state to store in the buffer
        :param action: the last action to store in the buffer
        :param reward: the reward obtained from the action at the state to store in the buffer
        :param value: the value of the state as estimated by the advantage stream of the model to store in the buffer
        :param args: additional arguments to store in the buffer, depending on the space type defined for it (e.g. expected-value, log-likelihood unmasked, etc)
        """
        # Append all data
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        # Parse the args depending on the action space type of the model
        # Note: they are parsed in the same order in which they are returned by the predict method of the model
        if self._action_space_type == SpaceType.discrete:
            self._log_likelihoods_unmasked.append(args[0])
        else:
            self._expected_values.append(args[0])
            self._log_stds.append(args[1])
        # Increase the pointer
        self._pointer += 1

    def finish_path(self,
                    value: float = 0):
        """
        Finish the path at the end of a trajectory. This looks back in the buffer to where the trajectory started,
        and uses rewards and value estimates from the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as the targets for the value function.

        :param value: the last reward given by the environment or the last predicted value if last state is not terminal
        """
        path_slice = slice(self._path_start_index, self._pointer)
        rewards: numpy.ndarray = numpy.array(self._rewards[path_slice] + [value])
        values: numpy.ndarray = numpy.array(self._values[path_slice] + [value])
        # Compute GAE-Lambda advantage estimation
        deltas: numpy.ndarray = numpy.concatenate(rewards[:-1] + self._discount_factor * values[1:] - values[:-1]).ravel()
        self._advantages[path_slice] = self._discount_cumulative_sum(deltas, self._discount_factor * self._lambda_parameter).tolist()
        # Compute rewards-to-go
        self._rewards_to_go[path_slice] = (self._discount_cumulative_sum(rewards, self._discount_factor)[:-1]).tolist()
        self._path_start_index = self._pointer

    def get(self) -> []:
        """
        Get all of the data from the buffer, with advantages appropriately normalized (shifted to have
        mean zero and standard deviation equals to one). Also reset pointers in the buffer and the list composing the buffer

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
        # Save the necessary values as ndarrays before reset
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
        # Save the additional values as ndarray depending on the action space type, then reset them and return all the arrays
        if self._action_space_type == SpaceType.discrete:
            log_likelihoods_unmasked_array: numpy.ndarray = numpy.array(self._log_likelihoods_unmasked)
            self._log_likelihoods_unmasked = []
            # Return all the buffer components with the required additional values
            return [states_array, actions_array, advantages_array, rewards_to_go_array, log_likelihoods_unmasked_array]
        else:
            expected_values_array: numpy.ndarray = numpy.array(self._expected_values)
            log_stds_array: numpy.ndarray = numpy.array(self._log_stds)
            self._expected_values = []
            self._log_stds = []
            # Return all the buffer components with the required additional values
            return [states_array, actions_array, advantages_array, rewards_to_go_array, expected_values_array, log_stds_array]

    @staticmethod
    def _discount_cumulative_sum(vector: numpy.ndarray, discount: float) -> numpy.ndarray:
        """
        Compute discounted cumulative sums of vectors.
        Credits to rllab.

        :param vector: the vector on which to compute cumulative discounted sum (e.g. [x0, x1, x2])
        :return the discounted cumulative sum (e.g. [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x3])
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], vector[::-1], axis=0)[::-1]


class TrustRegionPolicyOptimization(PolicyOptimizationModel):
    """
    TODO summary
    TODO: FINISH ALGORITHM

    Supported observation spaces:
        - discrete
        - continuous

    Supported action spaces:
        - discrete
        - continuous
    """

    def __init__(self,
                 name: str,
                 learning_rate: float,
                 discount_factor: float,
                 value_steps_for_update: int,
                 hidden_layers_config: Config,
                 lambda_parameter: float, damping_coefficient: float,
                 conjugate_gradient_iterations: int, kl_divergence_limit: float):
        # Define Trust Region Policy Optimization model attributes
        self._hidden_layers_config: Config = hidden_layers_config
        self.learning_rate: float = learning_rate
        self._value_steps_for_update: int = value_steps_for_update
        self._lambda_parameter: float = lambda_parameter
        self._damping_coefficient: float = damping_coefficient
        self._conjugate_gradient_iterations: int = conjugate_gradient_iterations
        self._kl_divergence_limit: float = kl_divergence_limit
        self._old_policies_log_likelihoods_unmasked = None
        self._log_likelihood_unmasked = None
        self._old_policies_expected_values = None
        self._old_policies_log_stds = None
        self._expected_value = None
        self._log_std = None
        self._kl_divergence = None
        self._old_policy_log_likelihood = None
        self._policy_ratio = None
        self._policy_gradient = None
        self._hessian_vector_product = None
        self._values = None
        self._get_actions_parameters = None
        self._set_actions_parameters = None
        # Generate the base policy optimization model
        super().__init__(name, discount_factor)
        # Define the types of allowed observation and action spaces
        self._supported_observation_space_types.append(SpaceType.discrete)
        self._supported_observation_space_types.append(SpaceType.continuous)
        self._supported_action_space_types.append(SpaceType.discrete)
        self._supported_action_space_types.append(SpaceType.continuous)

    def _define_graph(self):
        """
        Overridden method of Model class: check its docstring for further information.
        """
        # Define the GAE buffer for the vanilla policy gradient algorithm
        self.buffer: Buffer = Buffer(self.discount_factor, self._lambda_parameter)
        # Define the tensorflow model
        with tensorflow.variable_scope(self.scope + "/" + self.name):
            # Define inputs of the estimator as a float adaptable array with shape Nx(S) where N is the number of examples and (S) the shape of the state
            self._inputs = tensorflow.placeholder(shape=[None, *self.observation_space_shape], dtype=tensorflow.float32, name="inputs")
            # Define the estimator network hidden layers from the config
            hidden_layers_output = self._hidden_layers_config.apply_hidden_layers(self._inputs)
            # Define the targets for learning with the same NxA adaptable size
            self._targets = tensorflow.placeholder(shape=(None, *self.action_space_shape), dtype=tensorflow.float32, name="targets")
            # Change the model definition according to its action space type
            if self.action_space_type == SpaceType.discrete:
                # Define the logits as outputs of the deep neural network with shape NxA where N is the number of inputs, A is the action size when its type is discrete
                logits = tensorflow.layers.dense(hidden_layers_output, *self.action_space_shape, name="logits")
                # Define the actions on the first shape dimension as a squeeze on the samples drawn from a categorical distribution on the logits
                self._actions = tensorflow.squeeze(tensorflow.multinomial(logits=logits, num_samples=1), axis=1, name="actions")
                # Define the log likelihood according to the categorical distribution and also the unmasked version of it
                # The unmasked version is the one not "filtered" by the targets mask
                self._log_likelihood, self._log_likelihood_unmasked = PolicyOptimizationModel.get_categorical_log_likelihood(self._targets, logits)
                # Define the old policy unmasked log-likelihoods as a placeholder with adaptable shape Nx(A) where (A) is the shape of the action space
                self._old_policies_log_likelihoods_unmasked = tensorflow.placeholder(shape=(None, *self.action_space_shape), dtype=tensorflow.float32, name="old_policy_log_likelihood_unmasked")
                # Define the KL divergence on the unmasked log-likelihoods (current policy vs old policy)
                self._kl_divergence = PolicyOptimizationModel.get_categorical_kl(self._log_likelihood_unmasked, self._old_policies_log_likelihoods_unmasked)
            else:
                # Define the expected value as the output of the deep neural network with shape Nx(A) where N is the number of inputs, (A) is the action shape
                self._expected_value = tensorflow.layers.dense(hidden_layers_output, *self.action_space_shape, name="expected_value")
                # Define the log standard deviation
                self._log_std = tensorflow.get_variable(name="log_std", initializer=-0.5*numpy.ones(*self.action_space_shape, dtype=numpy.float32))
                # Define the standard deviation
                std = tensorflow.exp(self._log_std, name="std")
                # Define actions as the expected value summed up with a noise vector multiplied by the standard deviation
                self._actions = tensorflow.add(self._expected_value, tensorflow.multiply(tensorflow.random_normal(tensorflow.shape(self._expected_value)), std), name="actions")
                # Define the log likelihood according to the gaussian distribution
                self._log_likelihood = PolicyOptimizationModel.get_gaussian_log_likelihood(self._targets, self._expected_value, self._log_std)
                # Define the old policy expected value as a placeholder of adaptable shape Nx(A) where (A) is the shape of the action space
                self._old_policies_expected_values = tensorflow.placeholder(shape=(None, *self.action_space_shape), dtype=tensorflow.float32, name="old_policy_expected_value")
                # Define the old policy std as a placeholder of adaptable shape Nx(A) where (A) is the shape of the action space
                self._old_policies_log_stds = tensorflow.placeholder(shape=(None, *self.action_space_shape), dtype=tensorflow.float32, name="old_policy_log_std")
                # Define the KL divergence on the properties of both gaussian distributions (current policy vs old policy)
                self._kl_divergence = PolicyOptimizationModel.get_diagonal_gaussian_kl(self._expected_value, self._log_std, self._old_policies_expected_values, self._old_policies_log_stds)
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
            self._value_stream_optimizer = tensorflow.train.AdamOptimizer(self.learning_rate).minimize(self._value_stream_loss)
            # Define the advantages as an adaptable vector of floats (they are computed by the MLP and stored in the buffer)
            # Note: the model get the advantages from the buffer once computed using GAE on the values
            self._advantages = tensorflow.placeholder(shape=(None,), dtype=tensorflow.float32, name="advantages")
            # Define the old policy log-likelihood placeholder as a vector of floats
            self._old_policy_log_likelihood = tensorflow.placeholder(shape=(None, ), dtype=tensorflow.float32, name="old_policy_log_likelihood")
            # Define the ratio of the policies as the exponential of the subtraction of the log-likelihoods (new policy and old policy)
            self._policy_ratio = tensorflow.exp(self._log_likelihood - self._old_policy_log_likelihood, name="policy_ratio")
            # Define the loss as the mean of the rewards multiplied the ratio of the policies
            self._policy_stream_loss = -tensorflow.reduce_mean(self._advantages * self._policy_ratio, name="policy_loss")
            # Define a flat list of trainable variables under the action scope to use as parameters for the gradient
            actions_parameters = [variable for variable in tensorflow.trainable_variables("actions")]
            # Define the flat gradient for the policy loss optimization
            self._policy_gradient = tensorflow.concat([tensorflow.reshape(parameter, (-1, )) for parameter in tensorflow.gradients(xs=actions_parameters, ys=self._policy_stream_loss)], axis=0)
            # Define the values placeholder and the hessian vector product of the KL divergence on the actions trainable parameters
            self._values, self._hessian_vector_product = self._get_hessian_vector_product(self._kl_divergence, actions_parameters)
            # Apply the damping coefficient, if any
            if self._damping_coefficient > 0:
                self._hessian_vector_product += self._damping_coefficient * self._values
            # Define the get-action-parameters operation on the action parameters
            self._get_actions_parameters = tensorflow.concat([tensorflow.reshape(x, (-1,)) for x in actions_parameters], axis=0)
            # Define the set-action-parameters operation on the action parameters and the values
            splits = tensorflow.split(self._values, [int(numpy.prod(parameter.shape.as_list())) for parameter in actions_parameters])
            new_parameters = [tensorflow.reshape(parameter_new, parameter.shape) for parameter, parameter_new in zip(actions_parameters, splits)]
            self._set_actions_parameters = tensorflow.group([tensorflow.assign(parameter, parameter_new) for parameter, parameter_new in zip(actions_parameters, new_parameters)])
            # Define the initializer
            self.initializer = tensorflow.global_variables_initializer()

    def _define_summary(self):
        """
        Overridden method of Model class: check its docstring for further information.
        """
        with tensorflow.variable_scope(self.scope + "/" + self.name):
            # Define the summary operation for this graph with losses summaries
            self.summary = tensorflow.summary.merge([tensorflow.summary.scalar("policy_stream_loss", self._policy_stream_loss),
                                                     tensorflow.summary.scalar("value_stream_loss", self._value_stream_loss)])

    def predict(self,
                session,
                observation_current) -> []:
        """
        Overridden method of Model class: check its docstring for further information.
        """
        # Return a random action sample given the current state and depending on the observation space type
        # Also compute value estimate
        if self.observation_space_type == SpaceType.discrete:
            actions, value, log_likelihood_unmasked = session.run([self._actions, self._value, self._log_likelihood_unmasked],
                                                                  feed_dict={self._inputs: [numpy.identity(*self.observation_space_shape)[observation_current]]})
            # Return the predicted action (first one in the distribution) and the estimated value in the shape of a list
            # Also return the log-likelihood unmasked
            return [actions[0], value, log_likelihood_unmasked]

        else:
            actions, value, expected_value, log_std = session.run([self._actions, self._value, self._expected_value, self._log_std],
                                                                  feed_dict={self._inputs: [observation_current]})
            # Return the predicted action (first one in the distribution) and the estimated value in the shape of a list
            # Also return the expected value and the log-std
            return [actions[0], value, expected_value, log_std]

    def update(self,
               session,
               batch: []):
        """
        Overridden method of PolicyOptimizationModel class: check its docstring for further information.
        """
        # Unpack the batch in the training arrays for necessary values
        inputs, targets, values, rewards = batch[0], batch[1], batch[2], batch[3]
        # Generate a one-hot encoded version of the inputs if observation space type is discrete
        if self.observation_space_type == SpaceType.discrete:
            inputs_array = numpy.array(inputs).reshape(-1)
            inputs = numpy.eye(*self.observation_space_shape)[inputs_array]
        # Generate a one-hot encoded version of the targets if action space type is discrete
        if self.action_space_type == SpaceType.discrete:
            targets_array = numpy.array(targets).reshape(-1)
            targets = numpy.eye(*self.action_space_shape)[targets_array]

        # TODO: FINISH HERE, CHANGE NAME VALUES TO SOMETHING APPROPRIATE, finish comments
        # Unpack the additional values in the batch depending on the action space type

        if self.action_space_type == SpaceType.discrete:
            log_likelihoods_unmasked = batch[4]
            h_x = lambda values: session.run(self._hessian_vector_product,
                                             feed_dict={
                                                self._old_policies_log_likelihoods_unmasked: log_likelihoods_unmasked,
                                                self._values: values
                                             })
            policy_gradient, policy_loss_old, value_loss_old = session.run([self._policy_gradient, self._policy_stream_loss, self._value_stream_loss],
                                                                           feed_dict={
                                                                                self._inputs: inputs,
                                                                                self._targets: targets,
                                                                                self._rewards: rewards,
                                                                                self._advantages: values,
                                                                                self._old_policies_log_likelihoods_unmasked: log_likelihoods_unmasked
                                                                           })
        else:
            expected_values = batch[4]
            log_stds = batch[5]
            h_x = lambda values: session.run(self._hessian_vector_product,
                                             feed_dict={
                                                self._old_policies_expected_values: expected_values,
                                                self._old_policies_log_stds: log_stds,
                                                self._values: values
                                             })
            policy_gradient, policy_loss_old, value_loss_old = session.run([self._policy_gradient, self._policy_stream_loss, self._value_stream_loss],
                                                                           feed_dict={
                                                                                self._inputs: inputs,
                                                                                self._targets: targets,
                                                                                self._rewards: rewards,
                                                                                self._advantages: values,
                                                                                self._old_policies_expected_values: expected_values,
                                                                                self._old_policies_log_stds: log_stds,
                                                                           })
        # TODO: Add iterations, delta?!
        x = self._get_conjugate_gradient(h_x, policy_gradient, self._conjugate_gradient_iterations)
        alpha = numpy.sqrt(2 * self._kl_divergence_limit / (numpy.dot(x, h_x(x)) + 1e-8))
        old_params = session.run(self._get_actions_parameters)


        # Run the policy optimizer of the model in training mode
        session.run(self._policy_stream_optimizer,
                    feed_dict={
                        self._inputs: inputs,
                        self._targets: targets,
                        self._advantages: values
                    })
        # Run the value optimizer of the model in training mode for the required amount of steps
        for _ in range(self._value_steps_for_update):
            session.run(self._value_stream_optimizer,
                        feed_dict={
                            self._inputs: inputs,
                            self._advantages: values,
                            self._rewards: rewards
                        })
        # Compute the policy loss and value loss of the model after this sequence of training and also compute the summary
        policy_loss, value_loss, summary = session.run([self._policy_stream_loss, self._value_stream_loss, self.summary],
                                                       feed_dict={
                                                           self._inputs: inputs,
                                                           self._targets: targets,
                                                           self._rewards: rewards,
                                                           self._advantages: values
                                                       })
        # Return both losses and summary for the update sequence
        return policy_loss, value_loss, summary

    @staticmethod
    def keys_as_sorted_list(dict):
        return sorted(list(dict.keys()))

    @staticmethod
    def values_as_sorted_list(dict):
        return [dict[k] for k in keys_as_sorted_list(dict)]

    def test(self):
        # Prepare hessian func, gradient eval
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        Hx = lambda x: mpi_avg(sess.run(hvp, feed_dict={**inputs, v_ph: x}))
        g, pi_l_old, v_l_old = sess.run([gradient, pi_loss, v_loss], feed_dict=inputs)
        g, pi_l_old = mpi_avg(g), mpi_avg(pi_l_old)

        # Core calculations for TRPO or NPG
        x = cg(Hx, g)
        alpha = np.sqrt(2 * delta / (np.dot(x, Hx(x)) + EPS))
        old_params = sess.run(get_pi_params)

        def set_and_eval(step):
            sess.run(set_pi_params, feed_dict={v_ph: old_params - alpha * x * step})
            return mpi_avg(sess.run([d_kl, pi_loss], feed_dict=inputs))

        if algo == 'npg':
            # npg has no backtracking or hard kl constraint enforcement
            kl, pi_l_new = set_and_eval(step=1.)

        elif algo == 'trpo':
            # trpo augments npg with backtracking line search, hard kl
            for j in range(backtrack_iters):
                kl, pi_l_new = set_and_eval(step=backtrack_coeff ** j)
                if kl <= delta and pi_l_new <= pi_l_old:
                    logger.log('Accepting new params at step %d of line search.' % j)
                    logger.store_train(BacktrackIters=j)
                    break

                if j == backtrack_iters - 1:
                    logger.log('Line search failed! Keeping old params.')
                    logger.store_train(BacktrackIters=j)
                    kl, pi_l_new = set_and_eval(step=0.)

        # Value function updates
        for _ in range(train_v_iters):
            sess.run(train_vf, feed_dict=inputs)
        v_l_new = sess.run(v_loss, feed_dict=inputs)

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

    @staticmethod
    def _get_hessian_vector_product(func, parameters):
        """
        Compute the Hessian vector product Hx where H = gradient ** 2 is the hessian of the given function where the gradient
        of it is computed on the given parameters.

        :param func: the function for which to compute the Hessian vector product
        :param parameters: the parameters of the function used to compute the gradient
        :return: the placeholder values of Hx and the Hessian
        """
        # for H = grad**2 f, compute Hx
        # Define the flat gradient for the given function given the parameters
        gradient = tensorflow.concat([tensorflow.reshape(parameter, (-1, )) for parameter in tensorflow.gradients(xs=parameters, ys=func)], axis=0)
        # Define values as a placeholder with the shape of the gradient
        x = tensorflow.placeholder(shape=gradient.shape, dtype=tensorflow.float32)
        # Return the values placeholder and the hessian
        return x, tensorflow.concat([tensorflow.reshape(parameter, (-1, )) for parameter in tensorflow.gradients(xs=parameters, ys=tensorflow.reduce_sum(gradient * x))], axis=0)

    @staticmethod
    def _get_conjugate_gradient(a_x, b: numpy.ndarray, iterations: int):
        """
        Get the solution values of the system Ax = b with conjugate gradient algorithm.

        :param a_x: the Ax left side of the equation
        :param b: the b right side of the equation (an ndarray)
        :param iterations: the number of iterations for which to run conjugate gradient algorithm
        :return the vector values solution of the equation
        """
        # Initialize arrays
        # Note: r should be 'b - Ax(values)', but for values = 0, Ax(values) = 0. Change it if doing warm start.
        x: numpy.ndarray = numpy.zeros_like(b)
        r: numpy.ndarray = b.copy()
        p: numpy.ndarray = r.copy()
        # Start conjugate gradient algorithm
        r_dot_old = numpy.dot(r, r)
        for _ in range(iterations):
            z = a_x(p)
            alpha = r_dot_old / (numpy.dot(p, z) + 1e-8)
            x += alpha * p
            r -= alpha * z
            r_dot_new = numpy.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x
