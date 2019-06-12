# Import required src

from usienarl.environment import Environment
from usienarl.models.q_learning_model import QLearningModel


class Explorer:
    """
    Base class for all the exploration systems. It should not be used by itself, and should be extended instead.

    When an explorer is created, it should be assigned to the experiment in order to be used by the model,
    also assigned to the experiment.

    The exploration rate current value at each step is stored in the experiment, and it's not a variable of the
    explorer per-se.

    Note: the explorers are not used by policy optimization algorithms.

    Attributes:
        - exploration_rate_start_value: float defining the start value of the exploration rate
        - exploration_rate_end_value: float defining the end value of the exploration rate

    """

    def __init__(self,
                 exploration_rate_start_value: float, exploration_rate_end_value: float,
                 exploration_rate_value_decay: float):
        # Define explorer attributes
        self.exploration_rate_start_value: float = exploration_rate_start_value
        self.exploration_rate_end_value: float = exploration_rate_end_value
        self._exploration_rate_value_decay: float = exploration_rate_value_decay

    def get_action(self,
                   exploration_rate: float,
                   model: QLearningModel, environment: Environment, session, state_current: int) -> []:
        """
        Get the action according to the exploration rate, the model and the current state of the environment.

        Note: not all the explorers use all the provided parameters.

        :param exploration_rate: the current exploration rate value
        :param model: the model to use, if needed, to choose the action
        :param environment: the environment in which the model operates
        :param session: the session of tensorflow currently running
        :param state_current: the current state of the environment
        :return the action in the shape supported by the environment in the shape of a list
        """
        # Empty method, definition should be implemented on a child class basis
        return None

    def update_exploration_rate(self,
                                exploration_rate_current_value: float) -> float:
        """
        Update the given current exploration rate according to the defined decay value, and return back the updated
        exploration rate.

        :param exploration_rate_current_value: the current exploration rate value to update (usually, decay)
        :return: the updated current exploration rate
        """
        # Update the given exploration rate value according to its decay and pass it back to the caller experiment
        updated_exploration_rate: float = exploration_rate_current_value - self._exploration_rate_value_decay
        if updated_exploration_rate < self.exploration_rate_end_value:
            updated_exploration_rate = self.exploration_rate_end_value
        return updated_exploration_rate
