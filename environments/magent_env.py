from ctypes import c_int
import importlib
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces.box import Box
from supersuit import black_death_v3

# these should be defined in the enviroment instead
AGENT_NAME_DICT = {
    "adversarial_pursuit": ['predator', 'prey'],
    "custom_adversarial_pursuit": ['predator', 'prey'],
    "battle": ['red', 'blue'],
    "custom_battle": ['red', 'blue'],
    "battlefield": ['red', 'blue'],
    "custom_battlefield": ['red', 'blue'],
    "combined_arms": ['redmelee', 'redranged', 'bluemelee', 'blueranged'],
    "custom_combined_arms": ['redmelee', 'redranged', 'bluemelee', 'blueranged'],
    "gather": ['omnivore'],
    "custom_gather": ['omnivore'],
    "tiger_deer": ['deer', 'tiger'],
    "custom_tiger_deer": ['deer', 'tiger']
}

class MAgentEnv(ParallelEnv):
    def __init__(self, env_id: str, seed: int, **kwargs) -> None:
        scenario = importlib.import_module('environments.' + env_id)

        self.env = black_death_v3(scenario.env(**kwargs)).unwrapped

        self.scenario_name = env_id
        self.n_handles = len(self.env.handles)
        self.side_names = AGENT_NAME_DICT[env_id]
        self.env.reset(seed)

        self.state_space = self.env.state_space
        self.observation_spaces = {}
        for k in self.env.agents:
            obs_space = self.env.observation_spaces[k]
            self.observation_spaces.update({
                k: Box(np.min(obs_space.low), np.max(obs_space.high), [np.prod(obs_space.shape)], obs_space.dtype, seed)
            })
        self.action_spaces = {k: self.env.action_spaces[k] for k in self.env.agents}
        self.agents = self.env.agents
        self.n_agents_all = len(self.agents)

        self.handles = self.env.handles

        self.agent_ids = [self.env.env.get_agent_id(h) for h in self.handles]
        self.n_agents = [self.env.env.get_num(h) for h in self.handles]

        self.metadata = self.env.metadata
        self.max_cycles = self.env.max_cycles
        self.individual_episode_reward = {k: 0.0 for k in self.agents}

    def reset(self, seed=None, option=None):
        observations, infos = self.env.reset(seed, option)
        for agent_key in self.agents:
            self.individual_episode_reward[agent_key] = 0.0
            observations[agent_key] = observations[agent_key].reshape([-1])
        reset_info = {
            "infos": {},
            "individual_episode_rewards": self.individual_episode_reward
        }
        return observations, reset_info

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        for k, v in rewards.items():
            self.individual_episode_reward[k] += v
            observations[k] = observations[k].reshape([-1])
        step_info = {"infos": infos,
                    "individual_episode_rewards": self.individual_episode_reward}
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

    def get_num(self, handle: int | c_int):
        if not isinstance(handle, (int, c_int)):
            raise TypeError("Handle is not of type int")
        try:
            n = self.env.env.get_num(handle)
        except:
            n = len(self.get_ids(handle))
        return n

    def get_ids(self, handle: int | c_int):
        if not isinstance(handle, (int, c_int)):
            raise TypeError("Handle is not of type int")
        try:
            ids = self.env.env.get_agent_id(handle)
        except:
            agent_name = AGENT_NAME_DICT[self.scenario_name][handle.value]
            ids_handle = []
            for id, agent_key in enumerate(self.agents):
                if agent_name in agent_key:
                    ids_handle.append(id)
            ids = ids_handle
        return ids

    def get_agent_mask(self):
        if self.handles is None:
            return np.ones(self.n_agents_all, dtype=np.bool_)  # all alive
        else:
            mask = np.zeros(self.n_agents_all, dtype=np.bool_)  # all dead
            for handle in self.handles:
                try:
                    alive_ids = self.get_ids(handle)
                    mask[alive_ids] = True  # get alive agents
                except AttributeError("Cannot get the ids for alive agents!"):
                    return
        return mask

    def get_handles(self):
        if hasattr(self.env, 'handles'):
            return self.env.handles
        else:
            try:
                return self.env.env.get_handles()
            except:
                handles = [c_int(h) for h in range(self.n_handles)]
                return handles

