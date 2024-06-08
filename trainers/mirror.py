from argparse import Namespace
from typing import Callable
from agents import AGENT_REGISTRY
from environments.magent_env import MAgentEnv
import torch.nn as nn
import torch
import numpy as np
from trainers.base import Base_Trainer


class Mirror_Trainer(Base_Trainer):

    def __init__(self, agent_config: Namespace, env: MAgentEnv, trainers: list) -> None:
        self.agent_config = agent_config
        self.side_name = self.agent_config.side_name
        self.mirrored_side = self.agent_config.mirrored_side
        self.algorithm = self.agent_config.algorithm
        self.env = env
        self.nn_agents = self._make_agents()

        for trainer in trainers:
            if trainer.side_name == self.mirrored_side:
                self.mirrored_agents = trainer.nn_agents
                self.mirrored_trainer = trainer
                if len(self.mirrored_agents) != len(self.nn_agents):
                    raise Exception(f'Side to mirror: {self.mirrored_side} has a different number of agents')

        if not hasattr(self, 'mirrored_agents'):
            raise ValueError(f'Side to mirror: "{self.mirrored_side}" not found in trainer list {trainers}')

    def get_actions(self, observations, infos) -> dict[str, torch.Tensor]:
        side_obs = {agent_name: obs for agent_name, obs in observations.items() if self.side_name in agent_name}
        mirrored_obs = {
            agent_name.replace(self.side_name,
                               self.mirrored_side): obs for agent_name,
            obs in side_obs.items()
        }

        mirrored_actions = self.mirrored_trainer.get_actions(mirrored_obs, infos)
        actions = {
            agent_name.replace(self.mirrored_side,
                               self.side_name): action for agent_name,
            action in mirrored_actions.items()
        }

        # section for mappo
        if self.side_name in actions:
            actions[self.side_name] = {
                agent_name.replace(self.mirrored_side,
                                   self.side_name): action for agent_name,
                action in actions[self.side_name].items()
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
