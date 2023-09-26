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

        self.observation_spaces = {f'agent{i}': Box(low=0, high=1, shape=(2,)) for i in range(_num_agents)}
        self.action_spaces = {f'agent{i}': Discrete(4) for i in range(_num_agents)}

        self.fig, self.ax = plt.subplots(figsize=(5, 5))  # Initialize the figure and axes
        self.agent_scatter = self.ax.scatter([], [], color='blue', s=500)  # Initialize the scatter object for agents
        self.target_scatter = self.ax.scatter([], [], color='red', s=500)  # Initialize the scatter object for targets
        plt.ion()

    def reset(self):
        self.agent_positions = np.random.uniform(0, 1, size=(self._num_agents, 2))
        self.target_positions = np.random.uniform(0, 1, size=(self.num_targets, 2))
        self.visited_targets.fill(False)
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

            if np.any(self.agent_positions[i] == 0) or np.any(self.agent_positions[i] == 1):
                rewards[i] -= 1

            # Check if agent has reached a target
            for j in range(self.num_targets):
                distance_old = np.linalg.norm(old_position - self.target_positions[j])
                distance_new = np.linalg.norm(self.agent_positions[i] - self.target_positions[j])

                if distance_new < 0.1:
                    if not self.visited_targets[j]:
                        # Agent reached a target that has not been visited before
                        rewards[i] += 1
                        self.visited_targets[j] = True
                    else:
                        # Agent reached a target that has already been visited
                        # print("visiting target again...")
                        rewards[i] -= 0.5
                elif distance_old > distance_new:
                    # Agent is getting closer to a target
                    rewards[i] += 0.1
                elif distance_old < distance_new:
                    # Agent is getting further from a target
                    rewards[i] -= 0.5

            # Check for collisions with other agents
            for j in range(i+1, self._num_agents):
                if np.linalg.norm(self.agent_positions[i] - self.agent_positions[j]) < 0.1:
                    # Agents i and j have collided
                    # print("agents have colided!")
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
