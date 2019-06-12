# Import packages

import gym

# Import required src

from usienarl.environment import Environment, SpaceType


class OpenAIEnvironment(Environment):
    """
    Wrapper environment for any OpenAI environment.
    """

    def __init__(self,
                 name: str):
        # Init the base environment
        super().__init__(name)

    def setup(self):
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        # If the environment is already defined, close it first and then make it again (resetting it)
        if self._environment is not None:
            self._environment.close()
        self._environment = gym.make(self.name)

    def close(self,
              session=None):
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        self._environment.close()

    def reset(self,
              session=None):
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        return self._environment.reset()

    def step(self,
             action: int,
             session=None):
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        state_next, reward, episode_done, _ = self._environment.step(action)
        return state_next, reward, episode_done

    def render(self,
               session=None):
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        self._environment.render()

    def get_random_action(self,
                          session=None):
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        return self._environment.action_space.sample()

    @property
    def observation_space_type(self) -> SpaceType:
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        # Get the observation space type depending on the gym space type
        if isinstance(self._environment.observation_space, gym.spaces.Discrete):
            return SpaceType.discrete
        else:
            return SpaceType.continuous

    @property
    def observation_space_shape(self):
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        # Get the observation space size depending on the gym space type
        if isinstance(self._environment.observation_space, gym.spaces.Discrete):
            return (self._environment.observation_space.n, )
        else:
            return self._environment.observation_space.high.shape

    @property
    def action_space_type(self) -> SpaceType:
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        # Get the action space type depending on the gym space type
        if isinstance(self._environment.action_space, gym.spaces.Discrete):
            return SpaceType.discrete
        else:
            return SpaceType.continuous

    @property
    def action_space_shape(self):
        """
        Overridden method of Experiment class: check its docstring for further information.
        """
        # Get the action space size depending on the gym space type
        if isinstance(self._environment.action_space, gym.spaces.Discrete):
            return (self._environment.action_space.n, )
        else:
            return self._environment.action_space.high.shape

