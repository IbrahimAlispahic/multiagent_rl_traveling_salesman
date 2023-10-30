from Ma_Ts_Environment.ma_ts_environment.env.ma_ts_environment import MaTsEnvironment

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import traceback
from q_network import QNetwork
import torch.nn.functional as F

from replay_buffer import ReplayBuffer

num_actions = 4
env = MaTsEnvironment(_num_agents=2, num_targets=6, num_actions=num_actions)
env.reset()

# Initialize a SummaryWriter
writer = SummaryWriter()

# Initialize the policy networks for each agent
# agents positions (2 * _num_agents), 
# agents velocities (2 * _num_agents),
# targets positions (2 * num_targets), 
# distances from other agents _num_agents * (_num_agents - 1) / 2 NOT USING 
# distances from targets (_num_agents * num_targets), 
# visited state of targets (num_targets),
# agent actions (_num_agents)
agent_positions = 2 * env._num_agents
agent_velocities = 2 * env._num_agents
target_positions = 2 * env.num_targets
# dist_from_agents = env._num_agents * (env._num_agents - 1) / 2
dist_from_targets = env._num_agents * env.num_targets
input_dim = agent_positions + agent_velocities + target_positions + dist_from_targets + env.num_targets + env._num_agents
# loaded_network = torch.load('q_networks.pth')['agent0']
q_networks = {f'agent{i}': QNetwork(int(input_dim), num_actions) for i in range(env._num_agents)}

learning_rate = 0.001  # You can experiment with this value
optimizers = {f'agent{i}': optim.Adam(q_networks[f'agent{i}'].parameters(), lr=learning_rate) for i in range(env._num_agents)}

# Training loop
num_episodes = 10_000
# average_reward_threshold = 10  # You can experiment with this value
stop_training = False

# Initialize epsilon for epsilon-greedy exploration
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.9975
epsilon = epsilon_start

gamma = 0.9  # Discount factor

batch_size = 64
# Initialize the ReplayBuffer
replay_buffer = ReplayBuffer(200000)  # Adjust capacity as needed

try:
    for episode in range(num_episodes):

        if episode % 100 == 0:
            print("EPISODE ", episode)
        # Reset the environment and get the initial state
        state = env.reset()

        # Initialize lists to store states, actions, and rewards for each agent
        rewards = {f'agent{i}': [] for i in range(env._num_agents)}

        # Initialize variables to store the total reward and episode length
        total_reward = 0
        episode_length = 0
        max_episode_length = 400

        # Episode loop
        done = False

        while not done:
            actions = np.zeros(env._num_agents)
            state_tensor = torch.tensor(state)
            for i in range(env._num_agents):
                # Select an action for each agent based on its policy
                q_values = q_networks[f'agent{i}'](state_tensor)

                if random.random() < epsilon:  # With probability epsilon, choose a random action
                    actions[i] = random.choice(range(num_actions))
                else:  # Otherwise, choose the action that the agent thinks is the best
                    # Add the action to the action dictionary
                    actions[i] = torch.argmax(q_values).item()
            
            # Take a step in the environment
            next_state, reward, done, _ = env.step(actions)
            next_state_tensor = torch.tensor(next_state)

            # Store the transition in the replay buffer
            replay_buffer.push(state, actions, reward, next_state, done)

            if episode_length >= max_episode_length:
                done = True  # End the episode

            # Only update the network if the replay buffer contains enough samples
            if len(replay_buffer) > batch_size:
                # Sample a batch of transitions from the replay buffer
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
                for i in range(env._num_agents):
                    action_i = action_batch[:, i]
                    reward_i = reward_batch[:, i]
                    # Compute the Q-values for the current and next states
                    q_values = q_networks[f'agent{i}'](state_batch)
                    next_q_values = q_networks[f'agent{i}'](next_state_batch)

                    # Compute the target Q-values
                    max_next_q_values, _ = torch.max(next_q_values, dim=1)

                    # Compute the target Q-values
                    target_q_values = reward_i + (gamma * max_next_q_values * (1 - done_batch))

                    # Compute the loss
                    q_value = q_values.gather(1, action_i.unsqueeze(1))
                    loss = F.mse_loss(q_value, target_q_values.unsqueeze(1))

                    # Update the Q-network
                    optimizers[f'agent{i}'].zero_grad()
                    loss.backward()
                    optimizers[f'agent{i}'].step()

            # Update the state with the new states and actions
            state = next_state

            # Update the total reward and episode length
            total_reward += np.sum(reward)
            average_reward = total_reward / (episode + 1)
            # if average_reward > average_reward_threshold:
            #     stop_training = True
            #     break
            episode_length += 1

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        # Log the metrics to TensorBoard
        writer.add_scalar('Total Reward', total_reward, episode)
        writer.add_scalar('Average Reward', average_reward, episode)
        writer.add_scalar('Episode Length', episode_length, episode)

        if stop_training:
            print(f"Stopping training, average reward {average_reward} reached at episode {episode}")
            break
except Exception as e:
    # Print the full stack trace
    traceback.print_exc()
    print("Training interrupted, saving weights...")
    writer.close()
    torch.save(q_networks, 'q_networks_interrupted.pth')
    print("Weights saved to 'q_networks_interrupted.pth'")

# Close the SummaryWriter
writer.close()
torch.save(q_networks, 'q_networks.pth')
