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


class MFQ_Trainer(Base_Trainer):
    def __init__(self, agent_config: Namespace, env: MAgentEnv) -> None:
        self.agent_config = agent_config
        self.env = env
        self.epsilon = self.agent_config.start_greedy
        self.temperature = self.agent_config.temperature
        super().__init__(agent_config, env)

        if self.agent_config.model_dir_load:
            self.load_agents()

    def get_action_futures(self, observations, infos) -> dict[str, torch.Future]:
        def _action_future():
            torch.argmax(
                    nn_agent(
                        torch.tensor(observations[nn_agent.agent_name].flatten()).to(
                            self.agent_config.device
                        ),
                        # TODO: tutaj taki problem:
                        # ta funkcja get_action_futures jest uzywana zeby dostac akcje z obserwacji
                        # ale zeby agent dostal akcje z obserwacji, musi znac mean_actions innych agentów
                        # czy to aby napewno tak ma działać?
                        torch.tensor(infos[nn_agent.agent_name]['actions_mean'].flatten()).to(
                            self.agent_config.device
                        ),
                    ),
                    # dim=0,
            )
            

        action_futures = {}
        for nn_agent in self.nn_agents:
            if random.random() < self.epsilon:
                action_fut = torch.jit.fork(
                    lambda: torch.tensor(
                        self.env.action_space(nn_agent.agent_name).sample()
                    )
                )
            else:
                breakpoint()
                action_fut = torch.jit.fork(
                    _action_future
                )

            action_futures[nn_agent.agent_name] = action_fut

        return action_futures

    def get_boltzmann_policy(self, x):
        return F.softmax((x / self.temperature), dim=-1)

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
        def _update_mfq():
            data = nn_agent.rb.sample(self.agent_config.batch_size)

            with torch.no_grad():
                # TODO: add mean action to this mix! (add mean actions to the buffer)
                target_max, indices = nn_agent.target_network(data.next_observations, data.next_actions_mean).max(dim=1)
                breakpoint()
                pi = self.get_boltzmann_policy(target_max)
                v_mf = target_max * pi
                td_target = (
                    data.rewards.flatten()
                    + self.agent_config.gamma * v_mf * (1 - data.dones.flatten())
                )
            old_val = nn_agent(data.observations, data.actions_mean).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)
            breakpoint()

            if global_step % 1 == 0:
                writer.add_scalar(f"td_loss/{nn_agent.agent_name}", loss, global_step)
                writer.add_scalar(
                    f"q_values/{nn_agent.agent_name}",
                    old_val.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    f"reward/{nn_agent.agent_name}",
                    rewards[nn_agent.agent_name],
                    global_step,
                )
                writer.add_scalar(
                    f"epsilon_greedy/{nn_agent.agent_name}",
                    self.epsilon,
                    global_step,
                )

            # optimize the model
            nn_agent.optimizer.zero_grad()
            loss.backward()
            nn_agent.optimizer.step()

            # update target network
            if global_step % self.agent_config.target_network_train_period == 0:
                for target_network_param, q_network_param in zip(
                    nn_agent.target_network.parameters(), nn_agent.parameters()
                ):
                    target_network_param.data.copy_(
                        self.agent_config.tau * q_network_param.data
                        + (1.0 - self.agent_config.tau) * target_network_param.data
                    )

        # TODO: currently data is added to the replay buffer on each global_step
        # maybe add the whole episode to replay buffer each episode
        # and add episodic returns to each reward
        rb_futures = []
        for nn_agent in self.nn_agents:
            # TODO: the infos may need some refactoring
            infos['infos'][nn_agent.agent_name]['actions_mean'] = ...
            infos['infos'][nn_agent.agent_name]['actions_mean_next'] = ...
            rb_futures.append(
                torch.jit.fork(
                    nn_agent.rb.add,
                    observations[nn_agent.agent_name],
                    next_observations[nn_agent.agent_name],
                    actions[nn_agent.agent_name],
                    np.array(rewards[nn_agent.agent_name]),
                    np.array(terminations[nn_agent.agent_name]),
                    infos[nn_agent.agent_name],
                )
            )

        for fut in rb_futures:
            torch.jit.wait(fut)

        update_futures = []
        if global_step > self.agent_config.learning_start:
            if global_step % self.agent_config.train_period == 0:
                for nn_agent in self.nn_agents:
                    update_futures.append(torch.jit.fork(_update_mfq))

        for fut in rb_futures:
            torch.jit.wait(fut)

        # TODO: should optimizer.lr be updated?
        self.epsilon = (1 - (global_step / self.agent_config.running_steps)) * self.agent_config.start_greedy
        self.epsilon = 0 if self.epsilon < 0 else self.epsilon
    

    def save_agents(self):

        save_path = self.agent_config.model_dir_save + "/" + self.agent_config.side_name

        if (not os.path.exists(save_path)) and (not self.agent_config.test_mode):
            os.makedirs(save_path)

        for nn_agent in self.nn_agents:
            torch.save(
                {
                    "agent_config": nn_agent.agent_config,
                    "agent_name": nn_agent.agent_name,
                    "network_state_dict": nn_agent.network.state_dict(),
                    "target_network_state_dict": nn_agent.target_network.state_dict(),
                    "optimizer_state_dict": nn_agent.optimizer.state_dict(),
                },
                save_path + f"/{nn_agent.agent_name}.tar",
            )

    def load_agents(self):
        model_files = sorted(
            os.listdir(self.agent_config.model_dir_load),
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if len(model_files) < len(self.nn_agents):
            raise Exception(
                f"Model directory {self.agent_config.model_dir_load} has fewer files than needed"
            )

        for nn_agent, file_name in zip(self.nn_agents, model_files):
            model_path = self.agent_config.model_dir_load + f"/{file_name}"
            model_tar = torch.load(model_path, map_location=self.agent_config.device)
            nn_agent.network.load_state_dict(model_tar["network_state_dict"])
            if not self.agent_config.load_policy_only:
                nn_agent.target_network.load_state_dict(
                    model_tar["target_network_state_dict"]
                )

            if not self.agent_config.reset_optimizer:
                nn_agent.optimizer.load_state_dict(model_tar["optimizer_state_dict"])

            nn_agent.to(self.agent_config.device)
