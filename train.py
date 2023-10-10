from combined_network import CombinedNetwork
from policy_network import PolicyNetwork
from Ma_Ts_Environment.ma_ts_environment.env.ma_ts_environment import MaTsEnvironment

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import traceback

from replay_buffer import ReplayBuffer

env = MaTsEnvironment(_num_agents=2, num_targets=6)
env.reset()

# Initialize a SummaryWriter
writer = SummaryWriter()

# Initialize the policy networks for each agent
input_dim = 2 * env._num_agents + 2 * env.num_targets + env.num_targets + env._num_agents
# loaded_network = torch.load('policy_networks_1a_1t_03.pth')['agent0']
# combined_network = CombinedNetwork(loaded_network, loaded_network)

policy_networks = {f'agent{i}': PolicyNetwork(input_dim, 4) for i in range(env._num_agents)}
# policy_networks = {f'agent{i}': combined_network for i in range(env._num_agents)}
# policy_networks = torch.load('policy_networks_2a_6t_02_50k.pth')

learning_rate = 0.001  # You can experiment with this value
optimizers = {f'agent{i}': optim.Adam(policy_networks[f'agent{i}'].parameters(), lr=learning_rate) for i in range(env._num_agents)}

# Training loop
num_episodes = 50_000
# average_reward_threshold = 10  # You can experiment with this value
stop_training = False

# Initialize epsilon for epsilon-greedy exploration
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
epsilon = epsilon_start

batch_size = 64

try:
    for episode in range(num_episodes):

        if episode % 100 == 0:
            print("EPISODE ", episode)
        # Reset the environment and get the initial state
        state = np.concatenate((env.reset().flatten(), np.zeros(env._num_agents), env.target_positions.flatten(), env.visited_targets.astype(float)))

        # Initialize lists to store states, actions, and rewards for each agent
        log_probs = {f'agent{i}': [] for i in range(env._num_agents)}
        rewards = {f'agent{i}': [] for i in range(env._num_agents)}

        # Initialize variables to store the total reward and episode length
        total_reward = 0
        episode_length = 0
        max_episode_length = 1000

        # Episode loop
        done = False
        # replay_buffer = ReplayBuffer(10000)  # Adjust capacity as needed

        while not done:
            actions = {}
            # pozicije agenta (2 * 2 = 4), akcije za agente (2), pozicije ciljeva(2 * 5 = 10), posjecenost ciljeva (5) => 4+10+5+2 = 21
            state_tensor = torch.tensor(state)
            for i in range(env._num_agents):
                # Select an action for each agent based on its policy
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

            if episode_length >= max_episode_length:
                done = True  # End the episode

            # Save the joint rewards
            # joint_reward = np.sum(reward)
            for i in range(env._num_agents):
                rewards[f'agent{i}'].append(reward[i])
                # rewards[f'agent{i}'].append(joint_reward)

            # replay_buffer.push(state, actions, rewards, next_state, done)
            # Update the state with the new states and actions
            state = np.concatenate((next_state.flatten(), np.array(list(actions.values())), env.target_positions.flatten(), env.visited_targets.astype(float)))

            # Update the total reward and episode length
            # total_reward += joint_reward
            total_reward += np.sum(reward)
            average_reward = total_reward / (episode + 1)
            # if average_reward > average_reward_threshold:
            #     stop_training = True
            #     break
            episode_length += 1

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        
        # Update the policy networks
        for i in range(env._num_agents):
            policy_loss = []
            G = 0
            gamma = 0.8  # Discount factor
            for t in range(len(rewards[f'agent{i}'])):
                G = sum([gamma**i * r for i, r in enumerate(rewards[f'agent{i}'][t:])])
                policy_loss.append(-log_probs[f'agent{i}'][t] * G)
            policy_loss = torch.stack(policy_loss).sum()

            optimizers[f'agent{i}'].zero_grad()
            policy_loss.backward()
            optimizers[f'agent{i}'].step()

        # Update the policy networks
        # for i in range(env._num_agents):
        #     gamma = 0.8  # Discount factor
        #     if len(replay_buffer) > batch_size:
        #         states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        #         for batch_index in range(batch_size):
        #             G = 0
        #             policy_loss = []
        #             agent_rewards = rewards[batch_index][f'agent{i}']
        #             for t in range(len(agent_rewards)):
        #                 G = sum([gamma**i * r for i, r in enumerate(agent_rewards[t:])])
        #                 policy_loss.append(-log_probs[f'agent{i}'][t] * G)
        #             policy_loss = torch.stack(policy_loss).sum()

        #             optimizers[f'agent{i}'].zero_grad()
        #             policy_loss.backward(retain_graph=True)
        #             optimizers[f'agent{i}'].step()

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
    torch.save(policy_networks, 'policy_networks_interrupted.pth')
    print("Weights saved to 'policy_networks_interrupted.pth'")

# Close the SummaryWriter
writer.close()
torch.save(policy_networks, 'policy_networks.pth')
