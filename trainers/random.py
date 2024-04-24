from supersuit.multiagent_wrappers import black_death_v3
from trainers.base import Base_Trainer
from environments.magent_env import MAgentEnv
from argparse import Namespace
import os
import torch
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Random_Trainer(Base_Trainer):
    def __init__(self, agent_config: Namespace, env: MAgentEnv) -> None:
        self.agent_config = agent_config
        self.side_name = self.agent_config.side_name
        self.algorithm = self.agent_config.algorithm
        self.env = env
        self.nn_agents = self._make_agents()

    def get_actions(self, observations, infos) -> dict[str, torch.Tensor]:
        action_futures = {}
        for nn_agent in self.nn_agents:
            action_fut = torch.jit.fork(
                lambda: torch.tensor(
                    self.env.action_space(nn_agent.agent_name).sample()
                )
            )

            action_futures[nn_agent.agent_name] = action_fut

        actions = {
            agent_name: torch.jit.wait(fut)
            for agent_name, fut in action_futures.items()
        }

        return actions

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
        pass

    def save_agents(self, checkpoint=None):
        pass

    def load_agents(self):
        pass
