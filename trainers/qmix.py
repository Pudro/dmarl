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
import torch.optim as optim


class QMIX_Trainer(Base_Trainer):
    def __init__(self, agent_config: Namespace, env: MAgentEnv) -> None:
        self.agent_config = agent_config
        self.env = env
        self.epsilon = self.agent_config.start_greedy
        super().__init__(agent_config, env)

        # initialize mixing network
        self.qmixer = QMixer(self.agent_config, self)
        self.target_qmixer = QMixer(self.agent_config, self)

        if self.agent_config.model_dir_load:
            self.load_agents()


    def get_actions(self, observations, infos) -> dict[str, torch.Tensor]:
        action_futures = {}
        for nn_agent in self.nn_agents:
            if random.random() < self.epsilon:
                action_fut = torch.jit.fork(
                    lambda: torch.tensor(
                        self.env.action_space(nn_agent.agent_name).sample()
                    )
                )
            else:
                action_fut = torch.jit.fork(
                    torch.argmax,
                    nn_agent(
                        torch.tensor(observations[nn_agent.agent_name].flatten()).to(
                            self.agent_config.device
                        ),
                    ),
                    # dim=0,
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
        rb_futures = []
        for nn_agent in self.nn_agents:
            rb_futures.append(
                torch.jit.fork(
                    nn_agent.rb.add,
                    observations[nn_agent.agent_name],
                    next_observations[nn_agent.agent_name],
                    actions[nn_agent.agent_name].cpu(),
                    np.array(rewards[nn_agent.agent_name]),
                    np.array(terminations[nn_agent.agent_name]),
                    infos[nn_agent.agent_name],
                )
            )

        for fut in rb_futures:
            torch.jit.wait(fut)

        if global_step > self.agent_config.learning_start:
            if global_step % self.agent_config.train_period == 0:
                agent_q_vals = torch.zeros((len(self.nn_agents), self.agent_config.batch_size, self.env.action_spaces[f'{self.agent_config.side_name}_0'].n))
                target_agent_q_vals = torch.zeros((len(self.nn_agents), self.agent_config.batch_size, self.env.action_spaces[f'{self.agent_config.side_name}_0'].n))
                batch_rewards = torch.zeros((len(self.nn_agents), self.agent_config.batch_size, 1))
                batch_dones = torch.zeros((len(self.nn_agents), self.agent_config.batch_size, 1))
                for i, nn_agent in enumerate(self.nn_agents):
                    data = nn_agent.rb.sample(self.agent_config.batch_size)
                    with torch.no_grad():
                        target_agent_q = nn_agent.target_network(data.next_observations)
                        target_agent_q_vals[i] = target_agent_q
                    agent_q = nn_agent(data.observations)
                    agent_q_vals[i] = agent_q
                    batch_rewards[i] = data.rewards
                    batch_dones[i] = data.dones

                    writer.add_scalar(
                        f"q_values/{nn_agent.agent_name}",
                        agent_q.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        f"reward/{nn_agent.agent_name}",
                        rewards[nn_agent.agent_name],
                        global_step,
                    )



                # update mixer network
                q_tot = self.qmixer(agent_q_vals, torch.tensor(self.env.state()).to(self.agent_config.device).flatten())
                with torch.no_grad():
                    target_q_tot = self.target_qmixer(target_agent_q_vals, torch.tensor(self.env.state()).to(self.agent_config.device).flatten())
                # batch_rewards = batch_rewards.contiguous().view(self.agent_config.batch_size, len(self.nn_agents), 1)
                # batch_dones = batch_dones.contiguous().view(self.agent_config.batch_size, len(self.nn_agents), 1)
                # Calculate 1-step Q-Learning targets
                targets = batch_rewards + self.agent_config.gamma * (1 - batch_dones) * target_q_tot
                # Td-error
                td_error = (q_tot - targets.detach())

                loss = (td_error ** 2).sum()

                # Optimise
                self.qmixer.optimizer.zero_grad()
                self.target_qmixer.optimizer.zero_grad()

                for nn_agent in self.nn_agents:
                    nn_agent.optimizer.zero_grad()

                loss.backward()
                # nn.utils.clip_grad_norm_(self.qmixer.parameters(), self.agent_config.max_grad_norm)
                # nn.utils.clip_grad_norm_(self.target_qmixer.parameters(), self.agent_config.max_grad_norm)

                self.qmixer.optimizer.step()
                self.target_qmixer.optimizer.step()

                for nn_agent in self.nn_agents:
                    nn_agent.optimizer.step()

                self.greedy_decay(global_step, infos)

                for nn_agent in self.nn_agents:
                    if global_step % 1 == 0:
                        writer.add_scalar(f"loss/{nn_agent.agent_name}", loss, global_step)
                        writer.add_scalar(
                            f"epsilon_greedy/{nn_agent.agent_name}",
                            self.epsilon,
                            global_step,
                        )

            if global_step % self.agent_config.target_network_train_period == 0:
                for target_network_param, q_network_param in zip(
                    self.target_qmixer.parameters(), self.qmixer.parameters()
                ):
                    target_network_param.data.copy_(
                        self.agent_config.tau * q_network_param.data
                        + (1.0 - self.agent_config.tau) * target_network_param.data
                    )

                for nn_agent in self.nn_agents:
                    for target_network_param, q_network_param in zip(
                        nn_agent.target_network.parameters(), nn_agent.parameters()
                    ):
                        target_network_param.data.copy_(
                            self.agent_config.tau * q_network_param.data
                            + (1.0 - self.agent_config.tau) * target_network_param.data
                            )

    def save_agents(self, checkpoint=None):

        save_path = self.agent_config.model_dir_save + "/" + self.agent_config.side_name
        if checkpoint:
            save_path = save_path + "/" + str(checkpoint)

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
                    # these should be saved in one separate file
                    "qmixer_state_dict": self.qmixer.state_dict(),
                    "target_qmixer_state_dict": self.target_qmixer.state_dict(),
                    "qmixer_optimizer_state_dict": self.qmixer.optimizer.state_dict(),
                    "target_qmixer_optimizer_state_dict": self.target_qmixer.optimizer.state_dict()
                },
                save_path + f"/{nn_agent.agent_name}.tar",
            )

    def load_agents(self):
        all_files = os.listdir(self.agent_config.model_dir_load)
        model_files = [f for f in all_files if os.path.isfile(os.path.join(self.agent_config.model_dir_load, f))]
        model_files = sorted(
            model_files,
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if len(model_files) < len(self.nn_agents):
            raise Exception(
                f"Model directory {self.agent_config.model_dir_load} has fewer files than needed"
            )

        for i, (nn_agent, file_name) in enumerate(zip(self.nn_agents, model_files)):
            model_path = self.agent_config.model_dir_load + f"/{file_name}"
            model_tar = torch.load(model_path, map_location=self.agent_config.device)

            if i == 0: # load qmixers
                # load matching weights only
                qmixer_dict = self.qmixer.state_dict()
                qmixer_dict.update({k: v for k, v in model_tar["qmixer_state_dict"].items() if k in qmixer_dict and v.size() == qmixer_dict[k].size()})
                self.qmixer.load_state_dict(qmixer_dict)

                target_qmixer_dict = self.target_qmixer.state_dict()
                target_qmixer_dict.update({k: v for k, v in model_tar["target_qmixer_state_dict"].items() if k in target_qmixer_dict and v.size() == target_qmixer_dict[k].size()})
                self.target_qmixer.load_state_dict(target_qmixer_dict)

                if not self.agent_config.reset_optimizer:
                    print("Loading optimizer state_dict with different parameters not supported. Optimizer params are stored as an ordered list and loading right params only is hard.")
                    # qmixer_optimizer_dict = self.qmixer.optimizer.state_dict()
                    # qmixer_optimizer_dict.update({k: v for k, v in model_tar["qmixer_optimizer_state_dict"].items() if k in qmixer_optimizer_dict and v.size() == qmixer_optimizer_dict[k].size()})
                    # self.qmixer.optimizer.load_state_dict(qmixer_optimizer_dict)

                    # target_qmixer_optimizer_dict = self.target_qmixer.optimizer.state_dict()
                    # target_qmixer_optimizer_dict.update({k: v for k, v in model_tar["target_qmixer_optimizer_state_dict"].items() if k in target_qmixer_optimizer_dict and v.size() == target_qmixer_optimizer_dict[k].size()})
                    # self.target_qmixer.optimizer.load_state_dict(target_qmixer_optimizer_dict)


                self.qmixer.to(self.agent_config.device)
                self.target_qmixer.to(self.agent_config.device)

            nn_agent.network.load_state_dict(model_tar["network_state_dict"])
            if not self.agent_config.load_policy_only:
                nn_agent.target_network.load_state_dict(
                    model_tar["target_network_state_dict"]
                )

            if not self.agent_config.reset_optimizer:
                nn_agent.optimizer.load_state_dict(model_tar["optimizer_state_dict"])

            nn_agent.to(self.agent_config.device)


class QMixer(nn.Module):
    def __init__(self, agent_config, trainer):
        super(QMixer, self).__init__()

        self.agent_config = agent_config
        self.n_agents = len(trainer.nn_agents)
        self.state_dim = int(np.prod(trainer.env.state().shape))

        # qs are b x n_agents x action_dim
        # w1 could be 1 x action_dim x embed_dim
        # w1 could also be 1 x n_agents x embed_dim
        # pros of using actions_dim:
        # - If we decide to change agent number, we still can use the weights
        # preos of using n_agents:
        # - If we decide to change agent type, we still can use the weights
        # thus for our purpose we will use the actions_dim

        self.embed_dim = agent_config.mixing_embed_dim
        hypernet_embed_dim = self.agent_config.hypernet_embed_dim
        self.action_dim = trainer.env.action_spaces[f'{agent_config.side_name}_0'].n

        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(hypernet_embed_dim, self.embed_dim * self.action_dim))

        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(hypernet_embed_dim, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=self.agent_config.learning_rate, eps=1e-5)
        self.to(self.agent_config.device)


    def forward(self, agent_qs, state):
        bs = agent_qs.size(1)
        state = state.flatten()
        # agent_qs = agent_qs.contiguous().permute(1,0,2) # this might introduce a bug - we want this tensor to be batch invariant, otherwise the net might not transfer well?
        # First layer
        w1 = torch.abs(self.hyper_w_1(state))
        b1 = self.hyper_b_1(state)
        w1 = w1.view(1, self.action_dim, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu((agent_qs @ w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(state))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(state).view(-1, 1, 1)
        # Compute final output
        y = (hidden @ w_final) + v
        # Reshape and return
        # q_tot = y.view(bs, -1, 1)
        return y
