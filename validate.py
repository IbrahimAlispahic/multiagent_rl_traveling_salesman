import torch
from Ma_Ts_Environment.ma_ts_environment.env.ma_ts_environment import MaTsEnvironment
import numpy as np


# Load the saved policy networks
policy_networks = torch.load('policy_networks_single_agent_single_target.pth')

# Create a new environment
env = MaTsEnvironment(_num_agents=1, num_targets=1)
env.reset()

# Define a function to select actions based on the policy networks
def select_actions(state, policy_networks):
    actions = {}
    for i in range(env._num_agents):
        state_tensor = torch.tensor(np.concatenate((state[i], env.target_positions.flatten(), env.visited_targets.astype(float))))
        action_probs = policy_networks[f'agent{i}'](state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        actions[f'agent{i}'] = action.item()
    return actions

# Run the environment for a few episodes and render each episode
num_episodes = 20
for episode in range(num_episodes):
    print(f"EPISODE {episode}")
    state = env.reset()
    done = False
    while not done:
        actions = select_actions(state, policy_networks)
        state, _, done, _ = env.step(actions)
        env.render()
