# Import required src

from usienarl import Environment
from usienarl.models import TemporalDifferenceModel


class ExplorationPolicy:
    """
    Base class for all the exploration systems. It should not be used by itself, and should be extended instead.

    When an explorer is created, it should be assigned to the experiment in order to be used by the model,
    also assigned to the experiment.

    The exploration rate current value at each step is stored in the experiment, and it's not a variable of the
    explorer per-se.

    Note: the explorers are not used by policy optimization algorithms.

    Attributes:
        - exploration_rate_max: float defining the start value of the exploration rate
        - exploration_rate_min: float defining the end value of the exploration rate
        - exploration_rate_decay: float defining how much each the exploration rate reduces itself when it is updated
    """

    def __init__(self,
                 exploration_rate_max: float, exploration_rate_min: float,
                 exploration_rate_decay: float):
        # Define attributes
        self.exploration_rate_max: float = exploration_rate_max
        self.exploration_rate_min: float = exploration_rate_min
        self.exploration_rate_decay: float = exploration_rate_decay
        # Define internal attributes
        self._exploration_rate: float = None

    def initialize(self):
        """
        Reset the exploration policy to its starting state.
        """
        # Reset the exploration rate
        self._exploration_rate = self.exploration_rate_max

    def get_action(self,
                   exploration_rate: float,
                   model: TemporalDifferenceModel, environment: Environment, session, state_current: int) -> []:
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

    def update(self):
        """
        Update the exploration policy, changing the exploration rate current value by its decay factor.
        """
        # Update the given exploration rate value according to its decay factor
        self._exploration_rate = max(self._exploration_rate - self.exploration_rate_decay, self.exploration_rate_min)

    @property
    def exploration_rate(self) -> float:
        return self._exploration_rate
