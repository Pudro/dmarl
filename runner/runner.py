from typing import List
from agents.utils import print_summary
from environments import MAgentEnv
from torch.utils.tensorboard.writer import SummaryWriter
from argparse import Namespace
import numpy as np
import wandb
from tqdm.auto import tqdm
import time
import os
import shutil
from trainers import TRAINER_REGISTRY
import torch
import copy
import threading
import random
import yaml
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
                text = f.read().replace('\n', '  \n')
                self.writer.add_text("config", text)
            self.use_wandb = False
        else:
            self.use_wandb = True
            # TODO: implement wandb logging
            raise NotImplementedError("Wandb logging not implemented")

        self.frame_list = []

        self.trainers = self._make_trainers(self.config)
        self.all_agent_networks = [nn_agent for trainer in self.trainers for nn_agent in trainer.nn_agents]
        self.episodic_returns = {nn_agent.agent_name: 0.0 for nn_agent in self.all_agent_networks}
        self.last_episodic_returns = {nn_agent.agent_name: 0.0 for nn_agent in self.all_agent_networks}

        print_summary(self.trainers)

        if hasattr(config.env, 'run_battle_test') and config.env.run_battle_test:
            self.run_battle_test()
        else:
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

            if agent_config.algorithm == 'Mirror':
                trainers.append(TRAINER_REGISTRY[agent_config.algorithm](agent_config, self.env, trainers))
            else:
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
        last_test = 0
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

                if not self.env._parallel_env.agents:
                    terminations = {k: True for k in terminations.keys()}

                    if hasattr(self.config.env, 'win_reward'):
                        rewards = self.add_win_reward(rewards)

                self.add_rewards(rewards)

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
                last_test += 1

                # save checkpoint
                for trainer in self.trainers:
                    if not trainer.agent_config.test_mode and global_step % trainer.agent_config.model_checkpoint_period == 0:
                        trainer.save_agents(global_step)

            self.save_video(global_step)
            self.last_episodic_returns = self.episodic_returns.copy()
            self.save_episodic_returns(global_step)
            cycle_bar.close()
            episode += 1

            # test episodes if specified
            for trainer in self.trainers:
                if hasattr(trainer.agent_config, 'test_algorithm') and last_test >= trainer.agent_config.test_period:

                    self.run_test_episodes(global_step, trainer)
                    last_test = 0

        global_bar.close()
        self.finish()

    def run_test_episodes(self, global_step, trainer):
        trainer_copy = copy.deepcopy(trainer)
        trainer_copy.epsilon = 0
        test_config = copy.deepcopy(trainer_copy.agent_config)
        test_config.side_name = trainer_copy.agent_config.test_side
        test_config.model_dir_load = trainer_copy.agent_config.test_opponent_dir
        test_config.start_greedy = 0
        test_config.epsilon = 0
        test_config.end_greedy = 0
        test_config.test_mode = True

        env_copy = copy.deepcopy(self.env)

        test_trainer = TRAINER_REGISTRY[trainer_copy.agent_config.test_algorithm](test_config, env_copy)

        def execute_in_thread():
            test_episode_wins = 0
            for test_episode in range(trainer_copy.agent_config.test_episodes):
                seed = np.random.randint(1e6)
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                print(
                    f'\nRunning test episode {test_episode+1}/{trainer_copy.agent_config.test_episodes}\nAt step: {global_step}\n'
                )
                observations, infos = env_copy.reset(seed)
                while env_copy._parallel_env.agents:    # when episode ends, the list is empty
                    actions = {
                        agent_name: action for trainer in (trainer_copy,
                                                           test_trainer) for agent_name,
                        action in trainer.get_actions(observations,
                                                      infos).items()
                    }
                    (
                        next_observations,
                        rewards,
                        terminations,
                        truncations,
                        infos,
                    ) = env_copy.step(actions)

                    if not env_copy._parallel_env.agents:
                        # eliminate opposing team to win
                        side_handles = {side: handle for side, handle in zip(env_copy.side_names, env_copy.handles)}
                        alive_agents = {
                            side: len(env_copy._parallel_env.env.get_alive(handle)) for side,
                            handle in side_handles.items()
                        }

                        if alive_agents[trainer_copy.side_name] > 0 and alive_agents[
                                trainer_copy.agent_config.test_side] == 0:
                            test_episode_wins += 1

                    observations = next_observations

            winrate = test_episode_wins / trainer_copy.agent_config.test_episodes
            self.writer.add_scalar(
                f"test_episode_winrate/{trainer_copy.side_name}",
                winrate,
                global_step,
            )

            np.random.seed(trainer_copy.agent_config.seed)
            torch.manual_seed(trainer_copy.agent_config.seed)
            random.seed(trainer_copy.agent_config.seed)

        thread = threading.Thread(target=execute_in_thread)
        thread.start()

    def add_win_reward(self, rewards):
        # eliminate enemy team to win
        side_handles = {side: handle for side, handle in zip(self.env.side_names, self.env.handles)}
        alive_agents = {side: len(self.env._parallel_env.env.get_alive(handle)) for side, handle in side_handles.items()}

        if alive_agents[self.env.side_names[0]] > 0 and alive_agents[self.env.side_names[1]] == 0:
            side_rewards = {
                agent_name: reward + self.config.env.win_reward for agent_name,
                reward in rewards.items() if self.env.side_names[0] in agent_name
            }
            rewards.update(side_rewards)

        elif alive_agents[self.env.side_names[0]] == 0 and alive_agents[self.env.side_names[1]] > 0:
            side_rewards = {
                agent_name: reward + self.config.env.win_reward for agent_name,
                reward in rewards.items() if self.env.side_names[1] in agent_name
            }
            rewards.update(side_rewards)

        return rewards

    def log_win(self, global_step):
        # if hasattr(self, '

        # eliminate enemy team to win
        side_handles = {side: handle for side, handle in zip(self.env.side_names, self.env.handles)}
        alive_agents = {side: len(self.env._parallel_env.env.get_alive(handle)) for side, handle in side_handles.items()}

        if alive_agents[self.env.side_names[0]] > 0 and alive_agents[self.env.side_names[1]] == 0:
            self.writer.add_scalar(
                f"total_episode_wins/{self.env.side_names[0]}",
                1,
                global_step,
            )
            self.writer.add_scalar(
                f"total_episode_wins/{self.env.side_names[1]}",
                0,
                global_step,
            )
        elif alive_agents[self.env.side_names[0]] == 0 and alive_agents[self.env.side_names[1]] > 0:
            self.writer.add_scalar(
                f"total_episode_wins/{self.env.side_names[1]}",
                1,
                global_step,
            )
            self.writer.add_scalar(
                f"total_episode_wins/{self.env.side_names[0]}",
                0,
                global_step,
            )

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

    def run_battle_test(self):
        global_step = 0
        last_test = 0

        for seed_no in tqdm(range(self.config.env.battle_test_seeds), desc='Battle test', position=2, leave=False):
            episode = 0
            won_episodes = {self.env.side_names[0]: 0, self.env.side_names[1]: 0}
            for episode in tqdm(range(self.config.env.battle_test_episodes),
                                desc=f'Seed {seed_no}',
                                position=1,
                                leave=False):
                cycle = 0
                cycle_bar = tqdm(total=self.config.env.max_cycles, desc=f'Episode {episode}', position=0, leave=False)
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

                    if not self.env._parallel_env.agents:
                        terminations = {k: True for k in terminations.keys()}

                        if hasattr(self.config.env, 'win_reward'):
                            rewards = self.add_win_reward(rewards)

                            side_handles = {side: handle for side, handle in zip(self.env.side_names, self.env.handles)}
                            alive_agents = {
                                side: len(self.env._parallel_env.env.get_alive(handle)) for side,
                                handle in side_handles.items()
                            }

                            # win condition is who has more agents at the end of the episode
                            if alive_agents[self.env.side_names[0]] > alive_agents[self.env.side_names[1]]:
                                won_episodes[self.env.side_names[0]] += 1
                            elif alive_agents[self.env.side_names[0]] < alive_agents[self.env.side_names[1]]:
                                won_episodes[self.env.side_names[1]] += 1

                        # store winning side

                    self.add_rewards(rewards)

                    observations = next_observations

                    if episode % self.config.env.render_episode_period == 0:
                        self.add_frame()

                    cycle_bar.update()
                    cycle += 1
                    global_step += 1
                    last_test += 1

                self.save_video(global_step)
                self.last_episodic_returns = self.episodic_returns.copy()
                self.save_episodic_returns(global_step)
                cycle_bar.close()
                episode += 1

            self.writer.add_scalar(
                f"total_won_episodes/{self.env.side_names[0]}",
                won_episodes[self.env.side_names[0]],
                seed_no,
            )
            self.writer.add_scalar(
                f"total_won_episodes/{self.env.side_names[1]}",
                won_episodes[self.env.side_names[1]],
                seed_no,
            )

            self.writer.add_scalar(
                f"episode_winrate/{self.env.side_names[0]}",
                won_episodes[self.env.side_names[0]] / self.config.env.battle_test_episodes,
                seed_no,
            )
            self.writer.add_scalar(
                f"total_won_episodes/{self.env.side_names[1]}",
                won_episodes[self.env.side_names[1]] / self.config.env.battle_test_episodes,
                seed_no,
            )

        self.finish()
