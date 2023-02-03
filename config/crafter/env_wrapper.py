from typing import Optional

import numpy as np

from core.game import Game
from core.utils import arr_to_str


class CrafterWrapper(Game):
    def __init__(self, env, discount: float, cvt_string=True, target_goal: Optional[str] = None):
        """Atari Wrapper
        Parameters
        ----------
        env: Any
            another env wrapper
        discount: float
            discount of env
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
        """
        super().__init__(env, env.action_space.n, discount)
        self.target_goal = target_goal
        self.cvt_string = cvt_string
        self.last_achievements = None

    def legal_actions(self):
        return list(range(self.env.action_space.n))

    def get_max_episode_steps(self):
        return self.env.get_max_episode_steps()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.target_goal is not None:
            reward = 0
            last_value = 0 if self.last_achievements is None else self.last_achievements[self.target_goal]
            if info["achievements"][self.target_goal] > last_value:
                reward = 1

        self.last_achievements = info["achievements"]
        observation = observation.astype(np.uint8)

        observation[-1, :22, 0] = self.build_achievements_vector() * 255

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation, reward, done, info

    def build_achievements_vector(self):
        if self.last_achievements is None:
            achievements_vector = np.zeros(22)
            return achievements_vector
        else:
            achievements = list(self.last_achievements.items())
            achievements.sort(key=lambda x: x[0])  # sort by the key
            achievements_vector = np.array([x[1] for x in achievements])
            return achievements_vector

    def add_achievements_vector(self, observation):
        pass

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = observation.astype(np.uint8)
        observation[-1, :22, 0] = self.build_achievements_vector() * 255
        self.last_achievements = None

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation

    def get_achievements(self):
        return self.last_achievements

    def close(self):
        self.env.close()
