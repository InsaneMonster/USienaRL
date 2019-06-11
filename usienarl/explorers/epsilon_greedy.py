# Import packages

import numpy

# Import required src

from usienarl import QLearningModel, Environment, Explorer


class EpsilonGreedyExplorer(Explorer):
    """
    Simple epsilon greedy explorer. It choose a random state if a random number in the interval [0, 1] is
    inferior to the current value of the exploration rate and such rate is also greater than zero.
    """

    def __init__(self,
                 exploration_rate_start_value: float, exploration_rate_end_value: float,
                 exploration_rate_value_decay: float):
        # Generate the base explorer
        super().__init__(exploration_rate_start_value, exploration_rate_end_value, exploration_rate_value_decay)

    def get_action(self,
                   exploration_rate_current_value: float,
                   model: QLearningModel, environment: Environment, session, state_current: int) -> []:
        """
        Overridden method of Explorer class: check its docstring for further information.
        """
        # Choose an action according to the epsilon greedy approach
        if exploration_rate_current_value > 0 and numpy.random.rand(1) < exploration_rate_current_value:
            action = environment.get_random_action()
        else:
            action = model.predict(session, state_current)[0]
        return [action]
