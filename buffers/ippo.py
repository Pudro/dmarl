import numpy as np


class IPPO_Buffer:

    def __init__(self, env, agent_config):
        self.env = env
        self.agent_config = agent_config
        self.batch_size = agent_config.batch_size
        self.buffer_size = agent_config.buffer_size
        self.obs_shape = env.observation_space(self.agent_config.side_name + '_0').shape
        # agent_count = len([0 for a in self.env.agents if self.agent_config.side_name in a])
        # self.obs_shape = np.prod(env.observation_space(self.agent_config.side_name + '_0').shape) * agent_count
        self.action_dim = env.action_space(self.agent_config.side_name + '_0').shape
        self.idx = 0

        self.clear_memory()

    def generate_batches(self):
        n_states = len(self.obs)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return (np.array(self.obs),
                np.array(self.actions),
                np.array(self.probs),
                np.array(self.vals),
                np.array(self.rewards),
                np.array(self.dones),
                batches)

    def store_memory(self, obs, action, probs, vals, reward, done):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.probs[self.idx] = probs
        self.vals[self.idx] = vals
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx += 1

    def clear_memory(self):
        self.obs = np.zeros((self.buffer_size, np.prod(self.obs_shape)), dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.float32)    # only one action possible
        self.vals = np.zeros(self.buffer_size, dtype=np.float32)
        self.probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.idx = 0
