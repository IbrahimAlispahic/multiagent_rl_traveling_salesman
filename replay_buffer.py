import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    # def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)  # Unpack values from each sample in the batch

        # Convert to numpy arrays
        state = np.array(state)
        next_state = np.array(next_state)
        reward = np.array(reward)
        done = np.array(done)

        # Convert action to a dictionary where keys are agent names and values are actions
        action_dict = {f'agent{i}': actions[i] for i, actions in enumerate(zip(*action))}

        return state, action_dict, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
