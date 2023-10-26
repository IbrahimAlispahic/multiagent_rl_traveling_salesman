import torch
from Ma_Ts_Environment.ma_ts_environment.env.ma_ts_environment import MaTsEnvironment

num_agents = 2
num_targets= 6
num_actions = 4

# Load the saved policy networks
# policy_networks = torch.load(f'policy_networks_{num_agents}a_{num_targets}t_04_20k.pth')
policy_networks = torch.load(f'policy_networks_2a_6t_04_100k.pth')

# Create a new environment
env = MaTsEnvironment(_num_agents=num_agents, num_targets=num_targets, num_actions=num_actions)

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
num_episodes = 50
for episode in range(num_episodes):
    print(f"EPISODE {episode}")
    state = env.reset()
    done = False
    while not done:
        actions = select_actions(state, policy_networks)
        next_state, rewards, done, _ = env.step(actions)
        # print("REWARDS: ", rewards)
        state = next_state

        env.render()
