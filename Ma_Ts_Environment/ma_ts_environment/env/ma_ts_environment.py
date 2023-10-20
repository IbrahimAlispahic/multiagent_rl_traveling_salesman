from pettingzoo.utils import ParallelEnv
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt
import numpy as np
import scipy.special

class MaTsEnvironment(ParallelEnv):
    def __init__(self, _num_agents, num_targets):
        
        self._num_agents = _num_agents
        self.num_targets = num_targets
        self.agent_positions = np.zeros((_num_agents, 2))
        self.target_positions = np.zeros((num_targets, 2))
        self.visited_targets = np.zeros(num_targets, dtype=bool)
        self.target_claims = np.zeros(num_targets, dtype=int)

        self.observation_spaces = {f'agent{i}': Box(low=0, high=1, shape=(2,)) for i in range(_num_agents)}
        self.action_spaces = {f'agent{i}': Discrete(4) for i in range(_num_agents)}

        self.fig, self.ax = plt.subplots(figsize=(5, 5))  # Initialize the figure and axes
        self.agent_scatter = self.ax.scatter([], [], color='blue', s=500)  # Initialize the scatter object for agents
        self.target_scatter = self.ax.scatter([], [], color='red', s=500)  # Initialize the scatter object for targets
        plt.ion()

    def reset(self):
        self.agent_positions = np.random.uniform(0.1, 0.9, size=(self._num_agents, 2))  # Avoid spawning near the boundary
        self.target_positions = np.random.uniform(0.1, 0.9, size=(self.num_targets, 2))  # Avoid spawning near the boundary
        self.visited_targets.fill(False)
        self.target_claims.fill(-1)
        return np.hstack((self.agent_positions.flatten(),
                          self.target_positions.flatten(),
                          self.visited_targets.astype(float),
                          np.zeros(self._num_agents)))

    def _calculate_movement(self, action):
        if action == 0:   # move up
            return np.array([0, 0.1])
        elif action == 1: # move down
            return np.array([0, -0.1])
        elif action == 2: # move left
            return np.array([-0.1, 0])
        elif action == 3: # move right
            return np.array([0.1, 0])
        
    def _calculate_reward_for_boundary_collision(self, agent_position):
        if np.any(agent_position == 0) or np.any(agent_position == 1):
            return -1
        return 0
    
    def _calculate_reward_for_target_reach(self, agent_position, target_positions, visited_targets):
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
    
    def _calculate_directional_reward(self, old_position, new_position, closest_target_position, use_simple_reward):
        # Calculate the distance to the target
        distance_old = np.linalg.norm(old_position - closest_target_position)
        distance_new = np.linalg.norm(new_position - closest_target_position)

        if use_simple_reward:
            return 0.2 if distance_old > distance_new else -0.5
        else:
            # Calculate the reward based on distance (exponential function)
            reward_distance = np.exp(-distance_new) - np.exp(-distance_old)

            # Calculate the reward based on direction
            # Compute the direction of movement
            direction_of_movement = np.arctan2(new_position[1] - old_position[1], new_position[0] - old_position[0])
            # Compute the direction towards the target
            direction_to_target = np.arctan2(closest_target_position[1] - old_position[1], closest_target_position[0] - old_position[0])
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

    def step(self, actions):
        rewards = np.zeros(self._num_agents)
        use_simple_reward = True  # Set this to True to use the simple reward approach
        
        for i in range(self._num_agents):
            # Get action for the agent
            action = actions[f'agent{i}']

            # Determine movement based on action
            movement = self._calculate_movement(action)

            # Add a small negative reward for each step
            rewards[i] -= 0.01

            # Save old position
            old_position = self.agent_positions[i].copy()

            # Update agent's position

            self.agent_positions[i] = np.clip(self.agent_positions[i] + movement, 0, 1)

            # Calculate the reward for boundary collision
            rewards[i] += self._calculate_reward_for_boundary_collision(self.agent_positions[i])

            # Calculate the reward for reaching a target
            reward_target, self.visited_targets = self._calculate_reward_for_target_reach(self.agent_positions[i], self.target_positions, self.visited_targets)
            rewards[i] += reward_target

            # Calculate the reward for moving towards the closest unvisited target
            if not np.all(self.visited_targets):
                unvisited_target_indices = np.where(~self.visited_targets)[0]
                distances_to_unvisited_targets = np.linalg.norm(self.target_positions[unvisited_target_indices] - self.agent_positions[i], axis=1)
                sorted_indices = np.argsort(distances_to_unvisited_targets)
                closest_target_position = None
                for target_index in sorted_indices:
                    if self.target_claims[unvisited_target_indices[target_index]] == -1 or self.target_claims[unvisited_target_indices[target_index]] == i:
                        closest_target_position = self.target_positions[unvisited_target_indices[target_index]]
                        self.target_claims[unvisited_target_indices[target_index]] = i
                        break
                if closest_target_position is not None:
                    rewards[i] += self._calculate_directional_reward(old_position, self.agent_positions[i], closest_target_position, use_simple_reward)

            # Check for collisions with other agents
            for j in range(i+1, self._num_agents):
                if np.linalg.norm(self.agent_positions[i] - self.agent_positions[j]) < 0.1:
                    # Agents i and j have collided
                    rewards[i] -= 1
                    rewards[j] -= 1

        done = all(self.visited_targets)
        infos = {f'agent{i}': {} for i in range(self._num_agents)}
        next_state = np.hstack((self.agent_positions.flatten(),
                                self.target_positions.flatten(),
                                self.visited_targets.astype(float),
                                np.array(list(actions.values()))))
        return next_state, rewards, done, infos


    def render(self):
        # Update the data of the scatter objects
        self.agent_scatter.set_offsets(self.agent_positions)
        target_colors = ['green' if visited else 'red' for visited in self.visited_targets]
        self.target_scatter.set_offsets(self.target_positions)
        self.target_scatter.set_color(target_colors)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.pause(0.05)
        self.fig.canvas.draw()  # Update the figure
        # plt.grid()
        self.fig.canvas.flush_events()  # Clear the figure
