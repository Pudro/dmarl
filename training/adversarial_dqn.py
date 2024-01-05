from environments import custom_adversarial
from time import sleep
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
import torch.optim as optim
import random
from .args import Args

# env = adversarial_pursuit_v4.parallel_env(map_size=10, render_mode='human')
env = custom_adversarial.parallel_env(map_size=32, render_mode='rgb_array')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Training:
    def __init__(self) -> None:
        self.env = env
        self.agent_networks = [
            QNetworkAgent(env, agent).to(device) for agent in env.agents
        ]
        self.target_agent_networks = [
            QNetworkAgent(env, agent).to(device) for agent in env.agents
        ]

        for agent_network, target_network in zip(self.agent_networks, self.target_agent_networks):
            target_network.load_state_dict(agent_network.state_dict())

    def run(self, episodes=10):
        global_step = 0
        actions = {agent: env.action_space(agent).sample() for agent in env.agents} # random sample of actions
        for _ in range(episodes):
            observations, infos = env.reset()
            while env.agents: #run until the agents die? or unil the env stops (implementation details should be in MAgent2)
                epsilon = linear_schedule(Args.start_e, Args.end_e, int(Args.exploration_fraction * Args.total_timesteps), global_step)
                for network in self.agent_networks:
                    if random.random() < epsilon:
                        action = np.array([self.env.action_space(network.agent_id).sample() for _ in range(1)]) # range(envs.num_envs)
                    else:
                        q_values = network(torch.Tensor(observations[network.agent_id].flatten()).to(device))
                        action = np.atleast_1d(torch.argmax(q_values, dim=0).cpu().numpy())

                    actions[network.agent_id] = action

                next_observations, rewards, terminations, truncations, infos = env.step(actions)

                # TODO: summary writer
                # if 'final_info' in infos:
                #     for info in infos['final_info']:
                #         if info and 'episode' in info:
                #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                #             writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                #             writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
                for network in self.agent_networks:
                    real_next_observations = next_observations.copy()
                    for idx, trunc in enumerate(truncations):
                        if trunc and 'final_observation' in infos[network.agent_id]:
                            real_next_observations[network.agent_id][idx] = infos[network.agent_id]['final_observation'][idx]

                    network.rb.add(observations[network.agent_id],
                                   real_next_observations[network.agent_id],
                                   actions[network.agent_id],
                                   np.array(rewards[network.agent_id]),
                                   np.array(terminations[network.agent_id]),
                                   list(infos[network.agent_id]))

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                observations = next_observations


                for network, target_network in zip(self.agent_networks, self.target_agent_networks):
                    if global_step > Args.learning_starts:
                        if global_step % Args.train_frequency == 0:
                            data = network.rb.sample(Args.batch_size)
                            with torch.no_grad():
                                target_max, _ = target_network(data.next_observations).max(dim=1)
                                td_target = data.rewards.flatten() + Args.gamma * target_max * (1 - data.dones.flatten())
                            old_val = network(data.observations).gather(1, data.actions).squeeze()
                            loss = F.mse_loss(td_target, old_val)

                            # if global_step % 100 == 0:
                            #     writer.add_scalar("losses/td_loss", loss, global_step)
                            #     writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                            #     print("SPS:", int(global_step / (time.time() - start_time)))
                            #     writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                            # optimize the model
                            network.optimizer.zero_grad()
                            loss.backward()
                            network.optimizer.step()

                    # update target network
                    if global_step % Args.target_network_frequency == 0:
                        for target_network_param, q_network_param in zip(target_network.parameters(), network.parameters()):
                            target_network_param.data.copy_(
                                Args.tau * q_network_param.data + (1.0 - Args.tau) * target_network_param.data
                            )

                global_step += 1

        self.save_networks()

    def save_networks(self):
        for agent in self.agent_networks:
            torch.save(agent, f'./saved_networks/{agent.agent_id}_adversarial_ppo.pt')


class QNetworkAgent(nn.Module):
    def __init__(self, env, agent):
        super().__init__()
        self.agent_id = agent

        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_spaces[agent].shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_spaces[agent].n),
        )

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=2.5e-4, eps=1e-5)

        self.rb = ReplayBuffer(
            Args.buffer_size,
            env.observation_spaces[agent],
            env.action_spaces[agent],
            device,
            handle_timeout_termination=False,
        )


    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
