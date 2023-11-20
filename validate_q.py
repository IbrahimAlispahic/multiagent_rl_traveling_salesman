import torch
import numpy as np
import random
# from torch.utils.tensorboard import SummaryWriter

from Ma_Ts_Environment.ma_ts_environment.env.ma_ts_environment import MaTsEnvironment

num_agents = 4
num_targets= 20
num_actions = 4

# Load the saved q networks
q_networks = torch.load(f'q_networks_4a_20t_01_100k.pth')
# writer = SummaryWriter()

# Create a new environment
env = MaTsEnvironment(_num_agents=num_agents, num_targets=num_targets, num_actions=num_actions, size=1)

# Define a function to select actions based on the policy networks
def select_actions(state, q_networks):
    actions = np.zeros(env._num_agents)
    for i in range(env._num_agents):
        state_tensor = torch.tensor(state)
        q_values = q_networks[f'agent{i}'](state_tensor)
        # if random.random() < 0.1:  # With probability epsilon, choose a random action
        #     print("taking random action...")
        #     actions[i] = random.choice(range(num_actions))
        # else: 
        actions[i] = torch.argmax(q_values).item()
    return actions

# Run the environment for a few episodes and render each episode
num_episodes = 10_000
for episode in range(num_episodes):
    print(f"EPISODE {episode}")
    state = env.reset()
    done = False
    ep_length = 0
    while not done:
        ep_length += 1
        if ep_length > 400:
            print("OVER 400 episode!")
            print(actions)
            break
            # env.render()
        actions = select_actions(state, q_networks)
        next_state, rewards, done, _ = env.step(actions)
        # print("REWARDS: ", rewards)
        # print("ACTIONS: ", actions)
        state = next_state

        env.render()
    # writer.add_scalar('Episode Length', ep_length, episode)