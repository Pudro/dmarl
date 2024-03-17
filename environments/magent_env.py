import importlib
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces.box import Box
from supersuit import black_death_v3


class MAgentEnv(ParallelEnv):
    def __init__(self, env_id: str, seed: int, **kwargs) -> None:
        scenario = importlib.import_module("environments." + env_id)

        self.env = scenario.env(**kwargs).unwrapped

        self.scenario_name = env_id
        self.side_names = self.env.names

        self.state_space = self.env.state_space
        self.observation_spaces = {}
        for k in self.env.agents:
            obs_space = self.env.observation_spaces[k]
            self.observation_spaces.update(
                {
                    k: Box(
                        np.min(obs_space.low),
                        np.max(obs_space.high),
                        [np.prod(obs_space.shape)],
                        obs_space.dtype,
                        seed,
                    )
                }
            )
        self.action_spaces = {k: self.env.action_spaces[k] for k in self.env.agents}
        self.agents = self.env.agents
        self._parallel_env = self.env
        self.n_agents_all = len(self.agents)

        self.handles = self.env.handles

        self.agent_ids = [self.env.env.get_agent_id(h) for h in self.handles]
        self.n_agents = [self.env.env.get_num(h) for h in self.handles]

        self.metadata = self.env.metadata
        self.max_cycles = self.env.max_cycles
        self.individual_episode_reward = {k: 0.0 for k in self.agents}

        # wrap in black death after initialization
        self.env = black_death_v3(self.env)
        self.env.reset(seed)

    def reset(self, seed=None, option=None):
        observations, infos = self.env.reset(seed, option)
        for agent_key in self.agents:
            self.individual_episode_reward[agent_key] = 0.0
            observations[agent_key] = observations[agent_key].reshape([-1])
        reset_info = {
            "infos": {},
            "individual_episode_rewards": self.individual_episode_reward,
        }
        return observations, reset_info

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        for k, v in rewards.items():
            self.individual_episode_reward[k] += v
            observations[k] = observations[k].reshape([-1])
        step_info = {
            "infos": infos,
            "individual_episode_rewards": self.individual_episode_reward,
        }
        return observations, rewards, terminations, truncations, step_info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def state(self):
        try:
            return np.array(self.env.state())
        except:
            return None
