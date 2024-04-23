from argparse import Namespace
from typing import Callable
from agents import AGENT_REGISTRY
from environments.magent_env import MAgentEnv
import torch.nn as nn
import torch
import numpy as np


class Base_Trainer:
    def __init__(self, agent_config: Namespace, env: MAgentEnv) -> None:
        self.agent_config = agent_config
        self.side_name = self.agent_config.side_name
        self.algorithm = self.agent_config.algorithm
        self.env = env
        self.nn_agents = self._make_agents()
        self.greedy_decay = self._get_greedy_decay()

    def step(self):
        raise NotImplementedError

    def get_action_futures(self, observations, infos) -> dict[str, torch.Future]:
        raise NotImplementedError

    def get_actions(self, observations, infos) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def update_agents(
        self,
        global_step,
        actions,
        observations,
        next_observations,
        rewards,
        infos,
        terminations,
        writer,
    ):
        raise NotImplementedError

    def save_agents(self, checkpoint=None):
        raise NotImplementedError

    def load_agents(self):
        raise NotImplementedError

    def _make_agents(self):
        agent_list = []

        for agent_name in self.env.agents:
            stripped_agent_name = agent_name.split("_")[0]

            if self.side_name == stripped_agent_name:
                agent_list.append(
                    AGENT_REGISTRY[self.algorithm](
                        self.agent_config, self.env, agent_name
                    )
                )

        return nn.ModuleList(agent_list)

    def _get_greedy_decay(self) -> Callable:
        if not hasattr(self.agent_config, 'greedy_decay_type') or self.agent_config.greedy_decay_type == 'Linear':
            return self._linear_greedy
        elif self.agent_config.greedy_decay_type == 'Adaptive':
            return self._adaptive_greedy
        else:
            return self._linear_greedy

    def _linear_greedy(self, global_step, infos):
        epsilon_step = (self.agent_config.start_greedy - self.agent_config.end_greedy) / self.agent_config.greedy_decay_steps
        self.epsilon = max(self.agent_config.end_greedy, self.agent_config.start_greedy - epsilon_step * global_step)

    def _adaptive_greedy(self, global_step, infos):
        last_average_returns = np.mean(tuple(r for k, r in infos['last_episodic_returns'].items() if self.side_name in k))
        print(f'epsilon: {self.epsilon}')
        print(f'greedy_reward_threshold: {self.agent_config.greedy_reward_threshold}')
        print(f'Last average returns: {last_average_returns}')
        if self.epsilon > self.agent_config.end_greedy and last_average_returns >= self.agent_config.greedy_reward_threshold:
            self.epsilon = self.epsilon - self.agent_config.greedy_delta
            self.agent_config.greedy_reward_threshold = self.agent_config.greedy_reward_threshold + self.agent_config.greedy_reward_increment
