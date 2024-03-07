from typing import List
from environments import MAgentEnv
from torch.utils.tensorboard.writer import SummaryWriter
from agents import AGENT_REGISTRY
from gymnasium.spaces.box import Box
from pathlib import Path
from argparse import Namespace
import numpy as np
import socket
import wandb
import tqdm
import time
import os


class Runner:
    def __init__(self, config) -> None:
        self.env = self._make_env(config)
        self.env.reset()

        self.agent_networks = self._make_agents(config)

        self.config = config
        time_string = time.asctime().replace(" ", "").replace(":", "_")

        seed = f"seed_{config.seed}_"
        config.model_dir_load = config.model_dir
        config.model_dir_save = os.path.join(
            os.getcwd(), config.model_dir, seed + time_string
        )

        if (not os.path.exists(config.model_dir_save)) and (not config.test_mode):
            os.makedirs(config.model_dir_save)

        if config.logger == "tensorboard":
            log_dir = os.path.join(os.getcwd(), config.log_dir, seed + time_string)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        else:
            self.use_wandb = True

    # TODO: create a trainer for each of the agent types (side_name)
    # different trainers for different algorithms
    def _make_env(self, config: Namespace):

        return MAgentEnv(
            config.env.env_id,
            config.env.seed,
            minimap_mode=config.env.minimap_mode,
            max_cycles=config.env.max_cycles,
            extra_features=config.env.extra_features,
            map_size=config.env.map_size,
            render_mode=config.env.render_mode,
        )

    def _make_agents(self, config: Namespace) -> List:
        # TODO: check here if the side_name in the config matches the env
        # also check if the number of agent types is matching
        # TODO: also decide whether to use python list to store the agents or to make a class
        # will hold all agents insde a nn.ModuleList
        for agent_config in config.agents:
            if agent_config.side_name not in self.env.side_names:
                raise ValueError(
                    f"Agent side name: '{agent_config.side_name}' does not match the environment: {self.env.side_names}"
                )

        agent_list = []

        for agent_name in self.env.agents:
            stripped_agent_name = agent_name.split("_")[0]
            for agent_config in config.agents:
                if agent_config.side_name == stripped_agent_name:
                    agent_list.append(
                        AGENT_REGISTRY[agent_config.algorithm](agent_config, self.env, agent_name, config.env.device)
                    )
                else:
                    raise ValueError(
                        f"Agent name: '{stripped_agent_name}' not in config"
                    )

        return agent_list

    def finish(self):
        self.env.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

    def run(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
