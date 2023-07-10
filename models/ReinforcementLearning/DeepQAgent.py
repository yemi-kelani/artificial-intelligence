import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque

class DeepQAgent(nn.Module):
  def __init__(
      self,
      device,
      epsilon:float, 
      gamma:float,
      state_space:int, 
      action_space:int, 
      hidden_size:int = 100,
      dropout:float = 0.15,
      train_start:int = 50,
      batch_size:int = 64,
  ):
    
    super(DeepQAgent, self).__init__()

    self.epsilon = epsilon
    self.epsilon_min = 0.001 * epsilon
    self.epsilon_decay_rate = 0.999
    self.gamma = gamma
    self.state_space = state_space
    self.action_space = action_space
    self.train_start = train_start
    self.batch_size = batch_size
    self.memory = deque(maxlen=2000)
    self.device = device
    
    self.model = nn.Sequential(
        nn.Linear(state_space, hidden_size),
        nn.Dropout(dropout),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(dropout),
        nn.Tanh(),
        nn.Linear(hidden_size, action_space)
    )

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
    if len(self.memory) > self.train_start:
      if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay_rate

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
  
  def replay(self, optimizer, criterion):
    if len(self.memory) > self.train_start:

      batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
      for state, action, reward, next_state, done in batch:

        q_values = self.forward(state)
        q_values_next = self.forward(next_state)

        if done:
          q_values[action] = reward
        else:
          q_values[action] = reward + (self.gamma * torch.max(q_values_next))

        # optimize the model
        loss = criterion(self.forward(state), q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
      self.decay_epsilon()
  
  def save_model(self, path):
    torch.save(self.state_dict(), path)
    print(f"Model saved to {path}")

# The ONNX.js version of the model should contain only the
# bare necessities for the forward pass of full version
class DeepQAgentOnnxVersion(nn.Module):
  def __init__(
      self,
      device,
      epsilon:float, 
      gamma:float,
      state_space:int, 
      action_space:int, 
      hidden_size:int = 100,
      dropout:float = 0.15,
      train_start:int = 50,
      batch_size:int = 64,
  ):
    
    super(DeepQAgentOnnxVersion, self).__init__()
    
    self.epsilon = epsilon
    self.epsilon_min = 0.001 * epsilon
    self.epsilon_decay_rate = 0.999
    self.gamma = gamma
    self.state_space = state_space
    self.action_space = action_space
    self.train_start = train_start
    self.batch_size = batch_size
    self.memory = deque(maxlen=2000)
    self.device = device
    
    self.model = nn.Sequential(
        nn.Linear(state_space, hidden_size),
        nn.Dropout(dropout),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(dropout),
        nn.Tanh(),
        nn.Linear(hidden_size, action_space)
    )

  def forward(self, state):
    return self.model(torch.flatten(state))