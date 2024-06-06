from typing import List
from agents.utils import print_summary

from supersuit import black_death_v3
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
from tqdm import tqdm
import time
import os
import shutil
from trainers import TRAINER_REGISTRY
import torch

from trainers.base import Base_Trainer


class Runner:

    def __init__(self, config) -> None:
        self.env = self._make_env(config)
        self.env.reset()

        self.config = config

        time_string = time.asctime().replace(" ", "").replace(":", "_")
        curr = time.localtime()
        time_string = (str(curr.tm_year) + "-" + str(curr.tm_mon) + "-" + str(curr.tm_mday) + "_" + str(curr.tm_hour) +
                       ":" + str(curr.tm_min) + ":" + str(curr.tm_sec))

        for agent_config in self.config.agents:
            agent_config.model_dir_save = os.path.join(os.getcwd(), agent_config.model_dir_save, time_string)

        if self.config.env.logger == "tensorboard":
            log_dir = os.path.join(os.getcwd(), self.config.env.log_dir, time_string)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)
            shutil.copy2(config.env.config_file, log_dir)
            with open(config.env.config_file, "r") as f:
                text = f.read()
                self.writer.add_text("config", text)
            self.use_wandb = False
        else:
            self.use_wandb = True
            # TODO: implement wandb logging
            raise NotImplementedError("Wandb logging not implemented")

        self.frame_list = []

        self.trainers = self._make_trainers(self.config)
        self.all_agent_networks = [nn_agent for trainer in self.trainers for nn_agent in trainer.nn_agents]
        self.episodic_returns = {nn_agent.agent_name: float('-inf') for nn_agent in self.all_agent_networks}
        self.last_episodic_returns = {nn_agent.agent_name: float('-inf') for nn_agent in self.all_agent_networks}

        print_summary(self.trainers)
        self.run()

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
                    f"Agent side name: '{agent_config.side_name}' does not match the environment: {self.env.side_names}")

            trainers.append(TRAINER_REGISTRY[agent_config.algorithm](agent_config, self.env))

        return trainers

    def finish(self):
        self.env.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

        for trainer in self.trainers:
            if not trainer.agent_config.test_mode:
                trainer.save_agents()

    def run(self):
        global_step = 0
        episode = 0
        global_bar = tqdm(total=self.config.env.running_steps, desc='Global step')

        while global_step < self.config.env.running_steps:
            cycle = 0
            cycle_bar = tqdm(total=self.config.env.max_cycles, desc=f'Episode {episode}')
            observations, infos = self.env.reset()
            while self.env._parallel_env.agents:    # when episode ends, the list is empty
                actions = self.get_all_actions(observations, infos)
                (
                    next_observations,
                    rewards,
                    terminations,
                    truncations,
                    infos,
                ) = self.env.step(actions)
                infos['last_episodic_returns'] = self.last_episodic_returns
                infos['episode'] = episode
                # current_episode_rewards.append(rewards)
                # TODO: this should be handled by the writer
                self.add_rewards(rewards)

                if self.env.state() is None:
                    terminations = {k: True for k in terminations.keys()}

                for trainer in self.trainers:
                    # NOTE: might change so that futures are emited instead of handled inisde
                    if not trainer.agent_config.test_mode:
                        trainer.update_agents(
                            global_step,
                            actions,
                            observations,
                            next_observations,
                            rewards,
                            infos,
                            terminations,
                            self.writer,
                        )

                observations = next_observations

                if episode % self.config.env.render_episode_period == 0:
                    self.add_frame()

                global_bar.update()
                cycle_bar.update()
                cycle += 1
                global_step += 1

                # save checkpoint
                for trainer in self.trainers:
                    if not trainer.agent_config.test_mode and global_step % trainer.agent_config.model_checkpoint_period == 0:
                        trainer.save_agents(global_step)

            self.save_video(global_step)
            self.last_episodic_returns = self.episodic_returns.copy()
            self.save_episodic_returns(global_step)
            cycle_bar.close()
            episode += 1

        global_bar.close()
        self.finish()

    def get_all_actions(self, observations, infos) -> dict[str, torch.Tensor]:

        return {
            agent_name: action for trainer in self.trainers for agent_name,
            action in trainer.get_actions(observations,
                                          infos).items()
        }

    def save_video(self, global_step):
        if self.frame_list:
            vid_tensor = torch.stack(self.frame_list, dim=0).unsqueeze(0)
            self.writer.add_video(tag="render", vid_tensor=vid_tensor, global_step=global_step, fps=60)
            self.frame_list = []

    def add_frame(self):
        frame = torch.tensor(self.env.render()).permute(2, 0, 1)
        self.frame_list.append(frame)

    def add_rewards(self, rewards):
        for name, reward in rewards.items():
            self.episodic_returns[name] += reward

    def save_episodic_returns(self, global_step):

        for nn_agent in self.all_agent_networks:
            self.writer.add_scalar(
                f"episodic_returns/{nn_agent.agent_name}",
                self.episodic_returns[nn_agent.agent_name],
                global_step,
            )

        for agent_config in self.config.agents:
            side_average_returns = np.mean([r for a, r in self.episodic_returns.items() if agent_config.side_name in a])
            self.writer.add_scalar(
                f"average_episodic_returns/{agent_config.side_name}",
                side_average_returns,
                global_step,
            )

        for name in self.episodic_returns.keys():
            self.episodic_returns[name] = 0
