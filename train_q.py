"""
Training script for multi-agent reinforcement learning.
"""

import random
import traceback
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from Ma_Ts_Environment.ma_ts_environment.env.ma_ts_environment import MaTsEnvironment
# from q_network import QNetwork
from q_network1 import ModifiedQNetwork
from replay_buffer import ReplayBuffer


def initialize_networks(input_dim, num_actions, num_agents):
    """
    Initialize Q-networks and target networks.
    """
    q_networks = {
        f"agent_{i}": ModifiedQNetwork(input_dim, num_actions) for i in range(num_agents)
    }
    target_networks = {
        f"agent_{i}": ModifiedQNetwork(input_dim, num_actions) for i in range(num_agents)
    }
    for i in range(num_agents):
        target_networks[f"agent_{i}"].load_state_dict(
            q_networks[f"agent_{i}"].state_dict()
        )
    return q_networks, target_networks


def select_epsilon_greedy_action(state, q_network, epsilon, num_actions):
    """
    Select an action using epsilon-greedy policy.
    """
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    q_values = q_network(torch.tensor(state, dtype=torch.double))
    return torch.argmax(q_values).item()


def update_networks(
    replay_buffer,
    batch_size,
    q_networks,
    target_networks,
    optimizers,
    gamma,
    num_agents,
):
    """
    Update Q-networks based on a batch from the replay buffer.
    """
    if len(replay_buffer) > batch_size:
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = replay_buffer.sample(batch_size)
        for i in range(num_agents):
            update_network(
                q_networks[f"agent_{i}"],
                target_networks[f"agent_{i}"],
                optimizers[f"agent_{i}"],
                state_batch,
                action_batch[:, i],
                reward_batch[:, i],
                next_state_batch,
                done_batch,
                gamma,
            )


def update_network(
    q_network,
    target_network,
    optimizer,
    state_batch,
    action_batch,
    reward_batch,
    next_state_batch,
    done_batch,
    gamma,
):
    """
    Perform a single network update step.
    """
    if isinstance(state_batch, np.ndarray):
        state_batch = torch.tensor(state_batch, dtype=torch.float32)
    else:
        state_batch = state_batch.clone().detach()

    if isinstance(next_state_batch, np.ndarray):
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
    else:
        next_state_batch = next_state_batch.clone().detach()

    q_values = q_network(state_batch)
    next_q_values = target_network(next_state_batch)
    max_next_q_values = torch.max(next_q_values, dim=1)[0]
    target_q_values = reward_batch + (gamma * max_next_q_values * (1 - done_batch))
    q_value = q_values.gather(1, action_batch.unsqueeze(1))
    loss = F.mse_loss(q_value, target_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    """
    Main training loop.
    """
    num_agents = 1
    num_targets = 1
    num_actions = 4

    env = MaTsEnvironment(
        _num_agents=num_agents, num_targets=num_targets, num_actions=num_actions
    )

    agent_positions = 2 * num_agents
    agent_velocities = 2 * num_agents
    target_positions = 2 * num_targets
    # dist_from_agents = env._num_agents * (env._num_agents - 1) / 2
    dist_from_targets = num_agents * num_targets
    input_dim = (
        agent_positions
        + agent_velocities
        + target_positions
        + dist_from_targets
        + num_targets
        + num_agents
    )
    q_networks, target_networks = initialize_networks(
        input_dim, num_actions, num_agents
    )
    optimizers = {
        f"agent_{i}": optim.Adam(
            q_networks[f"agent_{i}"].parameters(), lr=0.001, weight_decay=0.01
        )
        for i in range(num_agents)
    }
    writer = SummaryWriter()
    replay_buffer = ReplayBuffer(1000000)
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9975
    epsilon = epsilon_start
    gamma = 0.8
    tau = 0.01
    num_episodes = 2_000
    batch_size = 64

    try:
        for episode in range(num_episodes):
            if episode % 200 == 0:
                print("EPISODE ", episode)

            state = env.reset()
            total_reward = 0
            episode_length = 0
            max_episode_length = 400
            done = False

            while not done:
                actions = np.zeros(num_agents)
                for i in range(num_agents):
                    action = select_epsilon_greedy_action(
                        state, q_networks[f"agent_{i}"], epsilon, num_actions
                    )
                    actions[i] = action

                next_state, reward, done, _ = env.step(actions)
                replay_buffer.push(state, actions, reward, next_state, done)
                update_networks(
                    replay_buffer,
                    batch_size,
                    q_networks,
                    target_networks,
                    optimizers,
                    gamma,
                    num_agents,
                )

                state = next_state
                total_reward += np.sum(reward)
                episode_length += 1
                if episode_length >= max_episode_length:
                    print("MAX LENGTH EPISODE!")
                    break

            for i in range(num_agents):
                for target_param, local_param in zip(
                    target_networks[f"agent_{i}"].parameters(),
                    q_networks[f"agent_{i}"].parameters(),
                ):
                    target_param.data.copy_(
                        tau * local_param.data + (1.0 - tau) * target_param.data
                    )

            epsilon = max(epsilon_end, epsilon_decay * epsilon)
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Average Reward", total_reward / (episode + 1), episode)
            writer.add_scalar("Episode Length", episode_length, episode)

    except Exception:
        traceback.print_exc()
        writer.close()
        torch.save(q_networks, "q_networks_interrupted.pth")

    writer.close()
    torch.save(q_networks, "q_networks.pth")


if __name__ == "__main__":
    main()
