import torch
from argparse import Namespace
from typing import Optional, Union
from environments.magent_env import MAgentEnv
from agents.base import Base_Agent

class IQL_Agent(Base_Agent):
    def __init__(
        self,
        agent_config: Namespace,
        env: MAgentEnv,
        agent_name: str,
        device: Optional[Union[str, int, torch.device]]
    ) -> None:
        self.agent_name = agent_name
        self.agent_config = agent_config
        self.env = env
        self.device = device

        super().__init__(agent_config, env, agent_name, device)

        raise NotImplementedError

