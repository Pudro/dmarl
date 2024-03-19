import torch
from argparse import Namespace
from typing import Optional, Union
from environments.magent_env import MAgentEnv
from agents.base import Base_Agent
import os
import copy


class IQL_Agent(Base_Agent):
    def __init__(
        self,
        agent_config: Namespace,
        env: MAgentEnv,
        agent_name: str,
    ) -> None:
        self.agent_name = agent_name
        self.agent_config = agent_config
        self.env = env
        self.device = self.agent_config.device

        super().__init__(agent_config, env, agent_name)
        self.target_network = copy.deepcopy(self.network)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.to(self.device)

    def save(self):

        save_path = self.agent_config.model_dir_save + '/' + self.agent_config.side_name
        if (not os.path.exists(save_path)) and (not self.agent_config.test_mode):
            os.makedirs(save_path)

        torch.save(
            {
                "agent_config": self.agent_config,
                "agent_name": self.agent_name,
                "network_state_dict": self.network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            save_path + f"/{self.agent_name}.tar",
        )
