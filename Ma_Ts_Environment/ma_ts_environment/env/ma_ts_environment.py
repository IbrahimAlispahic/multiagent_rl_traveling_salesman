from pettingzoo.utils import ParallelEnv
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt
import numpy as np

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
        return self.agent_positions

    def step(self, actions):
        rewards = np.zeros(self._num_agents)
        for i in range(self._num_agents):
            # Get action for the agent
            action = actions[f'agent{i}']

            # Determine movement based on action
            if action == 0:   # move up
                movement = np.array([0, 0.1])
            elif action == 1: # move down
                movement = np.array([0, -0.1])
            elif action == 2: # move left
                movement = np.array([-0.1, 0])
            elif action == 3: # move right
                movement = np.array([0.1, 0])

            # Add a small negative reward for each step
            rewards[i] -= 0.01

            # Save old position
            old_position = self.agent_positions[i].copy()

            # Update agent's position
            self.agent_positions[i] = np.clip(self.agent_positions[i] + movement, 0, 1)

            # Colliding with boundary of the environment
            if np.any(self.agent_positions[i] == 0) or np.any(self.agent_positions[i] == 1):
                rewards[i] -= 1

            # Check if agent has reached a target
            for j in range(self.num_targets):
                distance_old = np.linalg.norm(old_position - self.target_positions[j])
                distance_new = np.linalg.norm(self.agent_positions[i] - self.target_positions[j])

                if distance_new < 0.1:
                    if not self.visited_targets[j]:
                        # Agent reached a target that has not been visited before
                        rewards[i] += 20
                        self.visited_targets[j] = True
                        self.target_claims[j] = -1  # Reset the claim as the target is now visited
                    else:
                        # Agent reached a target that has already been visited
                        rewards[i] -= 0.5
            
            closest_target_position = None
            # Only consider unvisited targets
            if not np.all(self.visited_targets):
                unvisited_target_indices = np.where(~self.visited_targets)[0]
                distances_to_unvisited_targets = np.linalg.norm(self.target_positions[unvisited_target_indices] - self.agent_positions[i], axis=1)
                sorted_indices = np.argsort(distances_to_unvisited_targets)
                
                for target_index in sorted_indices:
                    if self.target_claims[unvisited_target_indices[target_index]] == -1 or self.target_claims[unvisited_target_indices[target_index]] == i:
                        closest_target_position = self.target_positions[unvisited_target_indices[target_index]]
                        self.target_claims[unvisited_target_indices[target_index]] = i
                        break

            # Calculate the reward based on whether the agent is getting closer to or further from the closest unvisited target
            if closest_target_position is not None:
                distance_old = np.linalg.norm(old_position - closest_target_position)
                distance_new = np.linalg.norm(self.agent_positions[i] - closest_target_position)
                if distance_old > distance_new:
                    # Agent is getting closer to the closest unvisited target
                    rewards[i] += 0.2
                elif distance_old < distance_new:
                    # Agent is getting further from the closest unvisited target
                    rewards[i] -= 0.5

            # Check for collisions with other agents
            for j in range(i+1, self._num_agents):
                if np.linalg.norm(self.agent_positions[i] - self.agent_positions[j]) < 0.1:
                    # Agents i and j have collided
                    rewards[i] -= 1
                    rewards[j] -= 1

        done = all(self.visited_targets)
        infos = {f'agent{i}': {} for i in range(self._num_agents)}
        return self.agent_positions, rewards, done, infos


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
