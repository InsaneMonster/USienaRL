# Import packages

import logging
import random

# Import required src

from usienarl import ExplorationPolicy, Interface, SpaceType


class EpsilonGreedyExplorer(ExplorationPolicy):
    """
    TODO: summary
    """

    def __init__(self,
                 exploration_rate_max: float, exploration_rate_min: float,
                 exploration_rate_decay: float):
        # Define epsilon-greedy exploration policy attributes
        self._exploration_rate_max: float = exploration_rate_max
        self._exploration_rate_min: float = exploration_rate_min
        self._exploration_rate_decay: float = exploration_rate_decay
        # Define epsilon-greedy empty exploration policy attributes
        self._exploration_rate: float = None
        # Generate the base exploration policy
        super().__init__()
        # Define the types of allowed action space types
        self._supported_action_space_types.append(SpaceType.discrete)
        self._supported_action_space_types.append(SpaceType.continuous)

    def _define(self):
        pass

    def initialize(self,
                   logger: logging.Logger,
                   session):
        # Reset exploration rate to its starting value (the max)
        self._exploration_rate = self._exploration_rate_max

    def act(self,
            logger: logging.Logger,
            session,
            interface: Interface,
            all_actions, best_action):
        # Choose an action according to the epsilon greedy approach: best action or random action
        if self._exploration_rate > 0 and random.uniform(0, 1.0) < self._exploration_rate:
            action = interface.get_random_agent_action(logger, session)
        else:
            action = best_action
        return action

    def update(self,
               logger: logging.Logger,
               session):
        # Decrease the exploration rate by its decay value
        self._exploration_rate = max(self._exploration_rate_min, self._exploration_rate - self._exploration_rate_decay)

