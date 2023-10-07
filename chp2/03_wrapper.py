import random
from typing import TypeVar

import gymnasium as gym
from gymnasium.core import Env

Action = TypeVar('Action')


class AgentActionWrapper(gym.ActionWrapper):
    def __init__(self, env: Env, epsilon: float = 0.1):
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print("Random action")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode='human')
    env = AgentActionWrapper(env)
    env.reset()

    total_reward = 0.0
    total_steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if terminated:
            break

    print(
        f"Episode done in {total_steps} steps, total reward {total_reward:.2f}")
