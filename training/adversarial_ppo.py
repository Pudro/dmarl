# from magent2.environments import adversarial_pursuit_v4
from environments import custom_adversarial
from time import sleep
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical

# env = adversarial_pursuit_v4.parallel_env(map_size=10, render_mode='human')
env = custom_adversarial.parallel_env(map_size=20, render_mode='human')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Training:
    def __init__(self) -> None:
        self.env = env
        self.agent_networks = [
            Agent(env, agent).to(device) for agent in env.agents
        ]

    def run(self, episodes=10, steps=100):
        for _ in range(episodes):
            observations, infos = env.reset()
            while env.agents: #run until the agents die? or unil the env stops (implementation details should be in MAgent2)
                actions = {agent: env.action_space(agent).sample() for agent in env.agents} # random sample of actions
                #actions = {agent: None for agent in env.agents}
                for network in self.agent_networks:
                    agent_observation = torch.from_numpy(observations[network.agent_id]).flatten()
                    action, logprob, entropy, value = network.get_action_and_value(agent_observation)
                    actions[network.agent_id] = action
                observations, rewards, terminations, truncations, infos = env.step(actions)


    def save_networks(self):
        for agent in self.agent_networks:
            torch.save(agent, 'adversarial_ppo.pt')

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
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

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

