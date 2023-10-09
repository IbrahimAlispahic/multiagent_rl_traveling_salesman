import torch
from Ma_Ts_Environment.ma_ts_environment.env.ma_ts_environment import MaTsEnvironment
import numpy as np


num_agents = 2
num_targets= 6

# Load the saved policy networks
# policy_networks = torch.load(f'policy_networks_{num_agents}a_{num_targets}t_04_20k.pth')
policy_networks = torch.load(f'policy_networks_2a_6t_01_50k.pth')

# Create a new environment
env = MaTsEnvironment(_num_agents=num_agents, num_targets=num_targets)

# Define a function to select actions based on the policy networks
def select_actions(state, policy_networks):
    actions = {}
    for i in range(env._num_agents):
        state_tensor = torch.tensor(state)
        action_probs = policy_networks[f'agent{i}'](state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        actions[f'agent{i}'] = action.item()
    return actions

# Run the environment for a few episodes and render each episode
num_episodes = 30
for episode in range(num_episodes):
    print(f"EPISODE {episode}")
    state = np.concatenate((env.reset().flatten(), np.zeros(env._num_agents), env.target_positions.flatten(), env.visited_targets.astype(float)))
    done = False
    while not done:
        actions = select_actions(state, policy_networks)
        next_state, _, done, _ = env.step(actions)
        state = np.concatenate((next_state.flatten(), np.array(list(actions.values())), env.target_positions.flatten(), env.visited_targets.astype(float)))

        env.render()
