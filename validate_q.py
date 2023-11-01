import torch
import numpy as np

from Ma_Ts_Environment.ma_ts_environment.env.ma_ts_environment import MaTsEnvironment

num_agents = 1
num_targets= 6
num_actions = 4

# Load the saved q networks
q_networks = torch.load(f'q_networks_1a_6t_04_10k.pth')

# Create a new environment
env = MaTsEnvironment(_num_agents=num_agents, num_targets=num_targets, num_actions=num_actions)

# Define a function to select actions based on the policy networks
def select_actions(state, q_networks):
    actions = np.zeros(env._num_agents)
    for i in range(env._num_agents):
        state_tensor = torch.tensor(state)
        q_values = q_networks[f'agent{i}'](state_tensor)
        actions[i] = torch.argmax(q_values).item()
    return actions

# Run the environment for a few episodes and render each episode
num_episodes = 100
for episode in range(num_episodes):
    print(f"EPISODE {episode}")
    state = env.reset()
    done = False
    while not done:
        actions = select_actions(state, q_networks)
        next_state, rewards, done, _ = env.step(actions)
        print("REWARDS: ", rewards)
        print("ACTIONS: ", actions)
        state = next_state

        env.render()
