from typing import List
from agents.base import Base_Agent
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
from trainers import TRAINER_REGISTRY
import torch

from trainers.base import Base_Trainer


class Runner:
    def __init__(self, config) -> None:
        self.env = self._make_env(config)
        self.env.reset()

        # self.agent_networks = self._make_agents(config)
        self.trainers = self._make_trainers(config)
        self.all_agent_networks = [
            nn_agent for trainer in self.trainers for nn_agent in trainer.nn_agents
        ]

        self.config = config
        # debug
        self.run()
        breakpoint()
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

    def _make_trainers(self, config: Namespace) -> List[Base_Trainer]:
        trainers = []
        for agent_config in config.agents:
            if agent_config.side_name not in self.env.side_names:
                raise ValueError(
                    f"Agent side name: '{agent_config.side_name}' does not match the environment: {self.env.side_names}"
                )

            trainers.append(
                TRAINER_REGISTRY[agent_config.algorithm](agent_config, self.env)
            )

        return trainers

    # def _make_agents(self, config: Namespace) -> List[Base_Agent]:
    #     # TODO: check here if the side_name in the config matches the env
    #     # also check if the number of agent types is matching
    #     for agent_config in config.agents:
    #         if agent_config.side_name not in self.env.side_names:
    #             raise ValueError(
    #                 f"Agent side name: '{agent_config.side_name}' does not match the environment: {self.env.side_names}"
    #             )
    #
    #     agent_list = []
    #
    #     for agent_name in self.env.agents:
    #         stripped_agent_name = agent_name.split("_")[0]
    #
    #         if stripped_agent_name not in (agent.side_name for agent in config.agents):
    #             raise ValueError(f"Agent name: '{stripped_agent_name}' not in config")
    #
    #         for agent_config in config.agents:
    #             if agent_config.side_name == stripped_agent_name:
    #                 agent_list.append(
    #                     AGENT_REGISTRY[agent_config.algorithm](
    #                         agent_config, self.env, agent_name, config.env.device
    #                     )
    #                 )
    #
    #     return agent_list

    def finish(self):
        self.env.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

    def run(self):
        for episode in range(1): # NOTE: debug
            cycle = 0
            observations, infos = self.env.reset()
            while cycle < self.config.env.max_cycles:
                action_futures = self.get_all_action_futures(observations)
                actions = {
                    agent_name: torch.jit.wait(fut) for agent_name, fut in action_futures.items()
                }
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                # current_episode_rewards.append(rewards)
                #
                if not self.config.env.test_mode: # skip if test mode
                    for network in self.all_agent_networks:
                        real_next_observations = next_observations.copy()
                        breakpoint()
                        # for idx, trunc in enumerate(truncations):
                        #     try:
                        #         if trunc and 'final_observation' in infos['infos'][network.agent_name]:
                        #             breakpoint()
                        #     except:
                        #         breakpoint()
                print(cycle)
                cycle += 1

        # raise NotImplementedError

    def get_all_action_futures(self, observations) -> dict[str, torch.Future]:
        # return [
        #     future for trainer in self.trainers for future in trainer.get_action_futures(observations)
        # ]

        return {
            agent_name: future
            for trainer in self.trainers
            for agent_name, future in trainer.get_action_futures(observations).items()
        }
