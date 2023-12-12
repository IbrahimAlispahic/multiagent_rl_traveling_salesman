from pettingzoo.utils import ParallelEnv
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import scipy.stats
from collections import deque
import random


class MaTsEnvironment(ParallelEnv):
    def __init__(self, _num_agents, num_targets, num_actions=4):
        self.fixed_agent_positions = None
        self.fixed_target_positions = None

        self.action_counts = np.zeros((_num_agents, num_actions))
        self.action_buffers = [
            deque(maxlen=20) for _ in range(_num_agents)
        ]  # Buffer for the last 10 actions for each agent

        self._num_agents = _num_agents
        self.num_targets = num_targets
        self.num_actions = num_actions
        self.agent_positions = np.zeros((_num_agents, 2))
        self.previous_agent_positions = np.zeros((_num_agents, 2))
        self.velocities = np.zeros((_num_agents, 2))
        self.target_positions = np.zeros((num_targets, 2))
        self.visited_targets = np.zeros(num_targets, dtype=bool)
        self.target_claims = np.zeros(num_targets, dtype=int)

        self.visited_locations = np.zeros((100, 100))  # Adjust the size as needed

        self.observation_spaces = {
            f"agent{i}": Box(low=0, high=1, shape=(2,)) for i in range(_num_agents)
        }
        self.action_spaces = {
            f"agent{i}": Discrete(num_actions) for i in range(_num_agents)
        }

        self.fig, self.ax = plt.subplots(
            figsize=(5, 5)
        )  # Initialize the figure and axes
        self.agent_scatter = self.ax.scatter(
            [], [], color="blue", s=500
        )  # Initialize the scatter object for agents
        self.target_scatter = self.ax.scatter(
            [], [], color="red", s=500
        )  # Initialize the scatter object for targets
        plt.ion()

    def reset(self):
        self.fixed_agent_positions = np.random.uniform(
            0.1, 0.9, size=(self._num_agents, 2)
        )  # Avoid spawning near the boundary
        self.fixed_target_positions = np.random.uniform(
            0.1, 0.9, size=(self.num_targets, 2)
        )  # Avoid spawning near the boundary

        self.agent_positions = self.fixed_agent_positions.copy()
        self.previous_agent_positions = self.agent_positions.copy()
        self.velocities = self.agent_positions - self.previous_agent_positions
        self.target_positions = self.fixed_target_positions.copy()
        self.visited_targets.fill(False)
        self.target_claims.fill(-1)
        # distances_to_other_agents = self.get_distances_to_other_agents()
        distances_to_targets = np.zeros((self._num_agents, self.num_targets))
        return np.hstack(
            (
                self.agent_positions.flatten(),
                self.velocities.flatten(),
                self.target_positions.flatten(),
                #   distances_to_other_agents,
                distances_to_targets.flatten(),
                self.visited_targets.astype(float),
                np.zeros(self._num_agents),
            )
        )

    def _calculate_movement(self, action):
        step_size = 0.01
        if action == 0:  # move up
            return np.array([0, step_size])
        elif action == 1:  # move down
            return np.array([0, -step_size])
        elif action == 2:  # move left
            return np.array([-step_size, 0])
        elif action == 3:  # move right
            return np.array([step_size, 0])
        elif action == 4:  # stay in place
            return np.array([0, 0])
        elif action == 5:  # move up and to the right (diagonal)
            return np.array([step_size, step_size])
        elif action == 6:  # move down and to the right (diagonal)
            return np.array([step_size, -step_size])
        elif action == 7:  # move down and to the left (diagonal)
            return np.array([-step_size, -step_size])
        elif action == 8:  # move up and to the left (diagonal)
            return np.array([-step_size, step_size])


    def _calculate_penalty_for_boundary_collision(self, agent_position):
        if np.any(agent_position == 0) or np.any(agent_position == 1):
            return 1
        return 0

    def _calculate_reward_for_target_reach(
        self, agent_position, target_positions, visited_targets
    ):
        reward = 0
        for j in range(self.num_targets):
            distance = np.linalg.norm(agent_position - target_positions[j])
            if distance < 0.1:
                if not visited_targets[j]:
                    reward += 20
                    visited_targets[j] = True
                else:
                    reward -= 0.5
        return reward, visited_targets

    def _calculate_penalty_for_visited_target(self, old_position, new_position):
        if np.any(self.visited_targets):
            visited_target_indices = np.where(self.visited_targets)[0]
            distances_to_visited_targets = np.linalg.norm(
                self.target_positions[visited_target_indices] - new_position, axis=1
            )
            closest_visited_target_index = np.argmin(distances_to_visited_targets)
            distance_old = np.linalg.norm(
                old_position
                - self.target_positions[visited_target_indices][
                    closest_visited_target_index
                ]
            )
            distance_new = distances_to_visited_targets[closest_visited_target_index]
            return 0.5 if distance_old > distance_new else 0
        return 0

    def _calculate_directional_reward(
        self, old_position, new_position, closest_target_position, use_simple_reward
    ):
        # Calculate the distance to the target
        distance_old = np.linalg.norm(old_position - closest_target_position)
        distance_new = np.linalg.norm(new_position - closest_target_position)

        if use_simple_reward:
            # Add a constant reward if getting closer to the target
            reward_closer = 0.2 if distance_old > distance_new else -0.5

            # If this is the last unvisited target, increase the penalty for moving away
            if np.sum(~self.visited_targets) == 1:
                reward_closer = 0.2 if distance_old > distance_new else -5
            return reward_closer
        else:
            # Calculate the reward based on distance (exponential function)
            reward_distance = np.exp(-distance_new) - np.exp(-distance_old)

            # Calculate the reward based on direction
            # Compute the direction of movement
            direction_of_movement = np.arctan2(
                new_position[1] - old_position[1], new_position[0] - old_position[0]
            )
            # Compute the direction towards the target
            direction_to_target = np.arctan2(
                closest_target_position[1] - old_position[1],
                closest_target_position[0] - old_position[0],
            )
            # Compute the difference between the two directions
            delta_direction = np.abs(direction_of_movement - direction_to_target)
            # Compute the reward for moving in the correct direction
            # using log function
            # reward_direction = -np.log(delta_direction + 1)
            # using sigmoid function
            reward_direction = scipy.special.expit(-delta_direction)

            # Add a constant reward if getting closer to the target
            reward_closer = 0.2 if distance_old > distance_new else 0

            # Combine the rewards
            reward = reward_distance + reward_direction + reward_closer

            # Add Gaussian noise to the reward
            # reward += np.random.normal(0, 0.1)

            return reward

    def _calculate_distances(self):
        distances = np.zeros((self._num_agents, self.num_targets))
        for i in range(self._num_agents):
            for j in range(self.num_targets):
                distances[i, j] = np.linalg.norm(
                    self.agent_positions[i] - self.target_positions[j]
                )
        return distances

    def get_distances_to_other_agents(self):
        # Expand dimensions for broadcasting
        agent_positions_expanded = np.expand_dims(self.agent_positions, axis=0)
        agent_positions_transposed = np.transpose(
            agent_positions_expanded, axes=(1, 0, 2)
        )

        # Calculate distances
        distances = np.linalg.norm(
            agent_positions_expanded - agent_positions_transposed, axis=2
        )

        # Set diagonal to large number to avoid selecting self in later steps
        np.fill_diagonal(distances, 0)
        n = distances.shape[0]

        upper_triangular_indices = np.triu_indices(n, k=1)
        upper_triangular_elements = distances[upper_triangular_indices]
        return upper_triangular_elements

    def detect_loop(self, buffer, agent_position):
        buffer_list = list(buffer)
        buffer_length = len(buffer_list)
        sequence_counts = {}
        for seq_length in range(
            2, 5
        ):  # We're looking for sequences of 2, 3 or 4 actions
            for start in range(buffer_length - seq_length * 2 + 1):
                seq = tuple(buffer_list[start : start + seq_length])
                # print("seq: ", seq)
                if (
                    len(set(seq)) > 1
                ):  # Check if not all actions in the sequence are the same
                    sequence_counts[seq] = sequence_counts.get(seq, 0) + 1
                    if sequence_counts[seq] >= 5:  # Detect a sequence repeating 5 times
                        return list(seq)
                else:
                    if (
                        agent_position[0] == 0
                        and seq[0] == 2
                        or agent_position[1] == 0
                        and seq[0] == 1
                        or agent_position[0] == 1
                        and seq[0] == 3
                        or agent_position[1] == 1
                        and seq[0] == 0
                    ):
                        return list(seq)

        return None

    def step(self, actions):
        actions = actions.astype(int)
         # Add some noise to the actions
        # noise = np.random.normal(0, 0.1, size=actions.shape)
        # actions = (actions + noise).clip(0, self.num_actions - 1).astype(int)

        rewards = np.zeros(self._num_agents)
        use_simple_reward = True  # Set this to True to use the simple reward approach

        for i in range(self._num_agents):
            # Get action for the agent
            action = actions[i]

            # # Add the action to the buffer
            # self.action_buffers[i].append(action)
            # # Check for loops
            # loop = self.detect_loop(self.action_buffers[i], self.agent_positions[i])
            # if loop is not None:
            #     # print("loop: ", loop)
            #     # If a loop is detected, select a random action that is different from the next action in the loop
            #     action = random.choice(
            #         [a for a in range(self.num_actions) if a != loop[0]]
            #     )
            #     rewards[i] -= 1
            #     # print("action: ", action)
            #     # print(loop)
            #     # print(self.action_buffers[i])
            #     # print("Loop detected, selecting a random action... ", action)

            # # Entropy reward
            # self.action_counts[i][action] += 1
            # # Normalize the action counts to get a distribution
            # action_distribution = self.action_counts[i] / np.sum(self.action_counts[i])
            # # Calculate the entropy
            # entropy = scipy.stats.entropy(action_distribution)
            # # Add the entropy to the reward
            # rewards[i] += 0.1 * entropy  # Adjust the coefficient as needed

            # Determine movement based on action
            movement = self._calculate_movement(action)

            # Add a small negative reward for each step
            rewards[i] -= 0.05

            # Save old position
            self.previous_agent_positions[i] = self.agent_positions[i].copy()

            # Update agent's position
            self.agent_positions[i] = np.clip(self.agent_positions[i] + movement, 0, 1)

            # Update visited locations and calculate the penalty
            x, y = (self.agent_positions[i] * (self.visited_locations.shape[0] - 1)).astype(int)  # Scale to grid size and subtract 1
            if self.visited_locations[x, y] == 1:
                rewards[i] -= 0.1  # Adjust the penalty as needed
            else:
                self.visited_locations[x, y] = 1

            # Calculate velocity
            self.velocities[i] = (
                self.agent_positions[i] - self.previous_agent_positions[i]
            )

            # Calculate the penalty for boundary collision
            rewards[i] -= self._calculate_penalty_for_boundary_collision(
                self.agent_positions[i]
            )

            # Calculate the reward for reaching a target
            (
                reward_target,
                self.visited_targets,
            ) = self._calculate_reward_for_target_reach(
                self.agent_positions[i], self.target_positions, self.visited_targets
            )
            rewards[i] += reward_target

            # Calculate the reward for moving towards the closest unvisited target
            if not np.all(self.visited_targets):
                unvisited_target_indices = np.where(~self.visited_targets)[0]
                distances_to_unvisited_targets = np.linalg.norm(
                    self.target_positions[unvisited_target_indices]
                    - self.agent_positions[i],
                    axis=1,
                )
                sorted_indices = np.argsort(distances_to_unvisited_targets)
                closest_target_position = None
                for target_index in sorted_indices:
                    if (
                        self.target_claims[unvisited_target_indices[target_index]] == -1
                        or self.target_claims[unvisited_target_indices[target_index]]
                        == i
                    ):
                        closest_target_position = self.target_positions[
                            unvisited_target_indices[target_index]
                        ]
                        self.target_claims[unvisited_target_indices[target_index]] = i
                        break
                if closest_target_position is not None:
                    rewards[i] += self._calculate_directional_reward(
                        self.previous_agent_positions[i],
                        self.agent_positions[i],
                        closest_target_position,
                        use_simple_reward,
                    )

            penalty = self._calculate_penalty_for_visited_target(
                self.previous_agent_positions[i], self.agent_positions[i]
            )
            rewards[i] -= penalty

            # Check for collisions with other agents
            for j in range(i + 1, self._num_agents):
                if (
                    np.linalg.norm(self.agent_positions[i] - self.agent_positions[j])
                    < 0.1
                ):
                    # Agents i and j have collided
                    rewards[i] -= 1
                    rewards[j] -= 1

        done = all(self.visited_targets)
        infos = {f"agent{i}": {} for i in range(self._num_agents)}
        distances_to_targets = self._calculate_distances()
        # distances_to_other_agents = self.get_distances_to_other_agents()
        next_state = np.hstack(
            (
                self.agent_positions.flatten(),
                self.velocities.flatten(),
                self.target_positions.flatten(),
                # distances_to_other_agents,
                distances_to_targets.flatten(),
                self.visited_targets.astype(float),
                actions,
            )
        )
        return next_state, rewards, done, infos

    def render(self):
        # Update the data of the scatter objects
        self.agent_scatter.set_offsets(self.agent_positions)
        target_colors = [
            "green" if visited else "red" for visited in self.visited_targets
        ]
        self.target_scatter.set_offsets(self.target_positions)
        self.target_scatter.set_color(target_colors)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.pause(0.1)
        self.fig.canvas.draw()  # Update the figure
        # plt.grid()
        self.fig.canvas.flush_events()  # Clear the figure
