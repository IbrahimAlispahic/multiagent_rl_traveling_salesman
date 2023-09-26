from policy_network import PolicyNetwork
from Ma_Ts_Environment.ma_ts_environment.env.ma_ts_environment import MaTsEnvironment

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random


env = MaTsEnvironment(_num_agents=2, num_targets=5)
env.reset()

# Initialize a SummaryWriter
writer = SummaryWriter()

# Initialize the policy networks for each agent
input_dim = 2 + 2 * env.num_targets + env.num_targets  
policy_networks = {f'agent{i}': PolicyNetwork(input_dim, 4) for i in range(env._num_agents)}
learning_rate = 0.001  # You can experiment with this value
optimizers = {f'agent{i}': optim.Adam(policy_networks[f'agent{i}'].parameters(), lr=learning_rate) for i in range(env._num_agents)}

# Training loop
num_episodes = 10_000
average_reward_threshold = 10  # You can experiment with this value
stop_training = False
try:
    for episode in range(num_episodes):
        print("EPISODE ", episode)
        # Reset the environment and get the initial state
        state = env.reset()

        # Initialize lists to store states, actions, and rewards for each agent
        log_probs = {f'agent{i}': [] for i in range(env._num_agents)}
        rewards = {f'agent{i}': [] for i in range(env._num_agents)}

        # Initialize variables to store the total reward and episode length
        total_reward = 0
        episode_length = 0

        # Episode loop
        done = False
        while not done:
            actions = {}
            epsilon = 0
            for i in range(env._num_agents):
                # Select an action for each agent based on its policy
                state_tensor = torch.tensor(np.concatenate((state[i], env.target_positions.flatten(), env.visited_targets.astype(float))))
                # single agent scenario
                # state_tensor = torch.tensor(np.concatenate((state[i], env.target_positions[0], [env.visited_targets[0].astype(float)])))
                action_probs = policy_networks[f'agent{i}'](state_tensor)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()

                # Save the log probability of the selected action
                log_probs[f'agent{i}'].append(action_dist.log_prob(action))

                if random.random() < epsilon:  # With probability epsilon, choose a random action
                    actions[f'agent{i}'] = random.choice(range(4))
                else:  # Otherwise, choose the action that the agent thinks is the best
                    # Add the action to the action dictionary
                    actions[f'agent{i}'] = action.item()
            
            # Take a step in the environment
            next_state, reward, done, _ = env.step(actions)

            # Save the rewards
            for i in range(env._num_agents):
                rewards[f'agent{i}'].append(reward[i])

            # Update the state
            state = next_state

            # Update the total reward and episode length
            total_reward += np.sum(reward)
            average_reward = total_reward / (episode + 1)
            # if average_reward > average_reward_threshold:
            #     stop_training = True
            #     break
            episode_length += 1

        # Update the policy networks
        for i in range(env._num_agents):
            policy_loss = []
            G = 0
            gamma = 0.99  # Discount factor
            for t in range(len(rewards[f'agent{i}'])):
                G = sum([gamma**i * r for i, r in enumerate(rewards[f'agent{i}'][t:])])
                policy_loss.append(-log_probs[f'agent{i}'][t] * G)
            policy_loss = torch.stack(policy_loss).sum()

            optimizers[f'agent{i}'].zero_grad()
            policy_loss.backward()
            optimizers[f'agent{i}'].step()

        # Log the metrics to TensorBoard
        writer.add_scalar('Total Reward', total_reward, episode)
        writer.add_scalar('Average Reward', average_reward, episode)
        writer.add_scalar('Episode Length', episode_length, episode)

        if stop_training:
            print(f"Stopping training, average reward {average_reward} reached at episode {episode}")
            break
except:
    print("Training interrupted, saving weights...")
    writer.close()
    torch.save(policy_networks, 'policy_networks_interrupted.pth')
    print("Weights saved to 'policy_networks_interrupted.pth'")

# Close the SummaryWriter
writer.close()
torch.save(policy_networks, 'policy_networks.pth')
