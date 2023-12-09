from Ma_Ts_Environment.ma_ts_environment.env.ma_ts_environment import MaTsEnvironment

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import traceback
from q_network import QNetwork
from policy_network import PolicyNetwork
import torch.nn.functional as F

from replay_buffer import ReplayBuffer

num_actions = 4
env = MaTsEnvironment(_num_agents=1, num_targets=1, num_actions=num_actions, size=1)
env.reset()

# Initialize a SummaryWriter
writer = SummaryWriter()

# Calculate input dimension for networks
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

# Initialize the policy networks and Q-networks for each agent
q_networks = {f'agent{i}': [QNetwork(int(input_dim), num_actions) for _ in range(2)] for i in range(env._num_agents)}  # Two Q-networks for each agent
policy_networks = {f'agent{i}': PolicyNetwork(int(input_dim), num_actions) for i in range(env._num_agents)}  # One policy network for each agent

# Initialize the target networks
target_q_networks = {f'agent{i}': [QNetwork(int(input_dim), num_actions) for _ in range(2)] for i in range(env._num_agents)}
target_policy_networks = {f'agent{i}': PolicyNetwork(int(input_dim), num_actions) for i in range(env._num_agents)}

# Copy the initial weights from the Q-networks and policy networks to the target networks
for i in range(env._num_agents):
    for j in range(2):
        target_q_networks[f'agent{i}'][j].load_state_dict(q_networks[f'agent{i}'][j].state_dict())
    target_policy_networks[f'agent{i}'].load_state_dict(policy_networks[f'agent{i}'].state_dict())


learning_rate = 0.001  # You can experiment with this value

# Defining optimizers
optimizers = {f'agent{i}': [optim.Adam(q_network.parameters(), lr=learning_rate) for q_network in q_networks[f'agent{i}']] for i in range(env._num_agents)}
policy_optimizers = {f'agent{i}': optim.Adam(policy_networks[f'agent{i}'].parameters(), lr=learning_rate) for i in range(env._num_agents)}

# Training loop
num_episodes = 5000
stop_training = False

policy_update = 2

# Initialize action_noise and noise_clip
action_noise = 0.1
noise_clip = 0.5

gamma = 0.8  # Discount factor
tau = 0.01  # Soft update rate
batch_size = 64

# Initialize the ReplayBuffer
replay_buffer = ReplayBuffer(200000)  # Adjust capacity as needed

try:
    for episode in range(num_episodes):

        if episode % 200 == 0:
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
                policy_output = policy_networks[f'agent{i}'](state_tensor)
                # Add noise to the action for exploration
                noise = torch.clamp(torch.randn_like(policy_output) * action_noise, -noise_clip, noise_clip)
                action = policy_output + noise
                action = torch.clamp(action, 0, 1)
                actions[i] = action.detach().numpy()[0]
            
            # Take a step in the environment
            next_state, reward, done, _ = env.step(actions)
            next_state_tensor = torch.tensor(next_state)

            # Store the transition in the replay buffer
            replay_buffer.push(state, actions, reward, next_state, done)

            if episode_length >= max_episode_length:
                print("MAX LENGTH EPISODE!")
                done = True  # End the episode

            # Only update the network if the replay buffer contains enough samples
            if len(replay_buffer) > batch_size:
                # Sample a batch of transitions from the replay buffer
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
                for i in range(env._num_agents):
                    action_i = action_batch[:, i]
                    reward_i = reward_batch[:, i].unsqueeze(1).detach()  # Unsqueeze to make it a column vector
                    done_i = done_batch.unsqueeze(1).detach()  # Unsqueeze to make it a column vector

                    # Compute the Q-values for the current and next states
                    q_values = [q_network(state_batch) for q_network in q_networks[f'agent{i}']]
                    # Compute the target Q-values using the target network
                    next_q_values = [target_q_network(next_state_batch.detach()) for target_q_network in target_q_networks[f'agent{i}']]
                    # Use the minimum of the two Q-values
                    min_next_q_values, _ = torch.min(torch.stack(next_q_values), dim=0)
                    min_next_q_values = min_next_q_values.squeeze(1)  # Make sure it's a 1D tensor
                    # Compute the target Q-values
                    target_q_values = reward_i + (gamma * min_next_q_values * (1 - done_i))
                    # Compute the losses
                    losses = [F.mse_loss(q_value.gather(1, action_i.unsqueeze(1)), target_q_values) for q_value in q_values]
                    # Update the Q-networks
                    for optimizer, loss in zip(optimizers[f'agent{i}'], losses):
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()

                    # Soft update the target networks
                    for target_q_network, q_network in zip(target_q_networks[f'agent{i}'], q_networks[f'agent{i}']):
                        for target_param, local_param in zip(target_q_network.parameters(), q_network.parameters()):
                            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
                    for target_param, local_param in zip(target_policy_networks[f'agent{i}'].parameters(), policy_networks[f'agent{i}'].parameters()):
                        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

                # Delayed policy updates
                if episode % policy_update == 0:
                    for i in range(env._num_agents):
                        # Compute the loss
                        q_values = q_networks[f'agent{i}'][0](state_batch)
                        action = policy_networks[f'agent{i}'](state_batch)
                        loss = -torch.mean(q_values.gather(1, action.unsqueeze(1)))
                        # Update the policy network
                        policy_optimizers[f'agent{i}'].zero_grad()
                        loss.backward()
                        policy_optimizers[f'agent{i}'].step()

            # Update the state with the new states and actions
            state = next_state
            # Update the total reward and episode length
            total_reward += np.sum(reward)
            average_reward = total_reward / (episode + 1)
            episode_length += 1

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
    torch.save(q_networks, 'td3_networks_interrupted.pth')
    print("Weights saved to 'td3_networks_interrupted.pth'")

# Close the SummaryWriter
writer.close()
torch.save(q_networks, 'td3_networks.pth')
