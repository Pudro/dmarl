from pettingzoo.mpe import simple_adversary_v3

env = simple_adversary_v3.parallel_env(render_mode="human")
observations, infos = env.reset()

class Training:
    def run(self, epochs=1):
        for _ in range(epochs):
            env.reset()
            while env.agents:
                # this is where you would insert your policy
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                observations, rewards, terminations, truncations, infos = env.step(actions)
        env.close()

