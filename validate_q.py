"""
Validation script for multi-agent reinforcement learning.
"""

import numpy as np
import torch
from torch.nn.functional import softmax
from Ma_Ts_Environment.ma_ts_environment.env.ma_ts_environment import MaTsEnvironment


def softmax_select_actions(
    state, q_networks, num_actions, use_softmax=True, temperature=1.0
):
    """
    Select actions based on Q values. If use_softmax is True, use softmax probabilities,
    otherwise select the action with the highest Q value.
    """
    actions = np.zeros(len(q_networks))
    for i, q_network in enumerate(q_networks.values()):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = q_network(state_tensor)

        if use_softmax:
            probabilities = softmax(q_values / temperature, dim=0).detach().numpy()
            action = np.random.choice(range(num_actions), p=probabilities)
        else:
            action = torch.argmax(q_values).item()

        actions[i] = action
    return actions


def main():
    """
    Main validation loop.
    """
    num_agents = 1
    num_targets = 1
    num_actions = 4
    env = MaTsEnvironment(
        _num_agents=num_agents, num_targets=num_targets, num_actions=num_actions
    )

    # Load the saved Q networks
    q_networks = torch.load("q_networks.pth")

    num_episodes = 5000
    for episode in range(num_episodes):
        print(f"EPISODE {episode}")
        state = env.reset()
        done = False
        ep_length = 0

        while not done:
            ep_length += 1
            if ep_length > 400:
                print("OVER 400 episode!")
                # env.render()
                # break

            actions = softmax_select_actions(state, q_networks, num_actions)
            next_state, _rewards, done, _ = env.step(actions)
            state = next_state
            env.render()

if __name__ == "__main__":
    main()
