from magent2 import gridworld
from magent2.environments import battlefield_v5
from magent2.environments import magent_env
import random
import numpy as np
import pygame.display
import magent2.gridworld
from time import sleep

from pettingzoo.utils.env import AECEnv
# By default, PettingZoo models games as Agent Environment Cycle (AEC) environments. This allows PettingZoo to represent any type of game multi-agent RL can consider.

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO


def random_demo(env: AECEnv, render: bool = True, episodes: int = 1) -> float:
    """Runs an env object with random actions."""
    total_reward = 0
    completed_episodes = 0

    while completed_episodes < episodes:
        env.reset()
        for agent in env.agent_iter():
            if render:
                env.render()

            obs, reward, termination, truncation, _ = env.last()
            total_reward += reward
            if termination or truncation:
                action = None
            elif isinstance(obs, dict) and "action_mask" in obs:
                action = random.choice(np.flatnonzero(obs["action_mask"]).tolist())
            else:
                action = env.action_space(agent).sample()
            env.step(action)

        completed_episodes += 1

    if render:
        env.close()

    print("Average total reward", total_reward / episodes)

    return total_reward

if __name__ == '__main__':
    env = battlefield_v5.parallel_env(render_mode='human', map_size=64, max_cycles=500)
    model = PPO('MlpPolicy', make_vec_env(env), verbose=1)
    random_demo(env, render=False, episodes=1)

