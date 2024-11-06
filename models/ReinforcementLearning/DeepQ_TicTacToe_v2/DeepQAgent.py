import os
import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque


class DeepQAgent(nn.Module):
    __name__      = "DeepQAgent"
    __author__    = "Yemi Kelani"
    __copyright__ = "Copyright (C) 2023, 2024 Yemi Kelani"
    __license__   = "Apache-2.0 license"
    __version__   = "2.0"
    __github__    = "https://github.com/yemi-kelani/artificial-intelligence"

    def __init__(
            self,
            device,
            epsilon: float,
            gamma: float,
            state_space: int,
            action_space: int,
            hidden_size: int = 256,
            dropout: float = 0.15,
            train_start: int = 100,
            batch_size: int = 256,
            negative_slope: float = 0.01,
            memory_max_len: int = 5000
            ):

        super(DeepQAgent, self).__init__()
        self.PT_EXTENSION = ".pt"

        self.device = device
        self.epsilon = epsilon
        self.epsilon_min = 0.001 * epsilon
        self.epsilon_decay_rate = 0.999
        self.gamma = gamma
        self.state_space = state_space
        self.action_space = action_space
        self.train_start = train_start
        self.batch_size = batch_size
        self.negative_slope = negative_slope
        self.memory = deque(maxlen=memory_max_len)
        self.memory_max_len = memory_max_len
        self.model = nn.Sequential(
            nn.Linear(state_space, hidden_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden_size, action_space)
        )
        self.model.to(device)
        self.init_weights()

    def init_weights(self):
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue

            if parameter.dim() == 1:
                # initialize bias
                nn.init.zeros_(parameter)

            if parameter.dim() > 1:
                # for tanh / sigmoid activation
                # nn.init.xavier_uniform_(parameter)

                # for leaky ReLU activations
                nn.init.kaiming_uniform_(
                    parameter,
                    a=self.negative_slope,
                    nonlinearity='leaky_relu')

    def forward(self, state):
        return self.model(torch.flatten(state))

    def select_action(self, state, mask, indicies):
        # select action greedily with respect to state
        # or select randomly from uniform distribution
        if np.random.rand() <= self.epsilon:
            return np.random.choice(indicies)
        else:
            q_values = self.forward(state)
            q_masked = torch.where(mask != 0, q_values, -1000)
            return torch.argmax(q_masked)

    def decay_epsilon(self):
        memory_length = len(self.memory)
        if memory_length >= self.train_start \
            or (self.memory_max_len > self.train_start \
                and memory_length == self.memory_max_len):
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_rate

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, optimizer, criterion):
        if len(self.memory) < self.train_start:
            return

        batch = random.sample(self.memory, min(
            len(self.memory), self.batch_size))
        for state, action, reward, next_state, done in batch:

            q_values = self.forward(state)
            q_values_next = self.forward(next_state)

            if done:
                q_values[action] = reward
            else:
                # Below is the Bellman equation. It stipulates that the
                # state-action (Q) value is its reward plus the max of the
                # rewards of the next state-action Q values.

                # Q(s, a) = reward + γ * Q(s', a')

                # γ is Gamma, the discount factor. This helps train the
                # model that rewards that are further away are less valuable.
                q_values[action] = reward + \
                    (self.gamma * torch.max(q_values_next))

            # optimize the model
            loss = criterion(self.forward(state), q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.decay_epsilon()

    def save_model(self, destination_path: str = "./", name: str = ""):
        if not os.path.exists(destination_path):
            os.makedirs(destination_path, exist_ok=True)
        filename = name if name.endswith(
            self.PT_EXTENSION) else name + self.PT_EXTENSION
        filepath = os.path.join(destination_path, filename)
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to '{filepath}'.")
        return filepath

    def load_model(self, filepath: str = ""):
        self.load_state_dict(torch.load(filepath, weights_only=True, map_location=self.device))
        print(f"Model loaded from '{filepath}'.")
