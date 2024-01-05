# from magent2.environments import adversarial_pursuit_v4
from environments import custom_adversarial
from time import sleep
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
import torch.optim as optim
from dataclasses import dataclass

# env = adversarial_pursuit_v4.parallel_env(map_size=10, render_mode='human')
env = custom_adversarial.parallel_env(map_size=20, render_mode='human')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Training:
    def __init__(self) -> None:
        self.env = env
        self.agent_networks = [
            PPOAgent(env, agent).to(device) for agent in env.agents
        ]

    def run(self, episodes=10, steps=100):
        dones = torch.zeros((1, 1)).to(device)
        next_done = torch.zeros(1).to(device)
        for _ in range(episodes):
            observations, infos = env.reset()
            while env.agents: #run until the agents die? or unil the env stops (implementation details should be in MAgent2)
                actions = {agent: env.action_space(agent).sample() for agent in env.agents} # random sample of actions
                for network in self.agent_networks:
                    with torch.no_grad():
                        agent_observation = torch.from_numpy(observations[network.agent_id]).flatten()
                        action, logropb, _, value = network.get_action_and_value(agent_observation)
                        value = value.flatten()
                    actions[network.agent_id] = action

                observations, rewards, terminations, truncations, infos = env.step(actions)

                # bootstrap value if not done
                for network in self.agent_networks:
                    with torch.no_grad():
                        agent_observation = torch.from_numpy(observations[network.agent_id]).flatten()
                        next_value = network.get_value(agent_observation).reshape(1, -1)

                        agent_reward = torch.from_numpy(np.array(rewards[network.agent_id])).flatten()
                        advantages = torch.zeros_like(agent_reward).to(device)

                        lastgaelam = 0
                        for t in reversed(range(1)):
                            if t == 1 - 1:
                                nextnonterminal = 1.0 - next_done
                                nextvalues = next_value
                            else:
                                nextnonterminal = 1.0 - dones[t + 1]
                                nextvalues = values[t + 1]
                            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                        returns = advantages + values



    def save_networks(self):
        for agent in self.agent_networks:
            torch.save(agent, 'adversarial_ppo.pt')

# this is so the layers are initialiazed as in the paper
# default torch initialization is different
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, env, agent):
        super().__init__()
        self.agent_id = agent
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_spaces[agent].shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_spaces[agent].shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_spaces[agent].n), std=0.01),
        )
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=2.5e-4, eps=1e-5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

