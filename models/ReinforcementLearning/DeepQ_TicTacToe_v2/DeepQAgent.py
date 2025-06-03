import os
import re
import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


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
            batch_size: int = 128,
            memory_max_len: int = 2000,
            use_target_network: bool = True,
            network_sync_rate: int = 10
    ):

        super(DeepQAgent, self).__init__()
        self.PT_EXTENSION = ".pt"
        self.LOG_DETAILS = False

        self.device = device
        self.epsilon = epsilon
        self.epsilon_min = 0.001 * epsilon
        self.epsilon_max = 1.0
        self.epsilon_decay_rate = 0.999
        self.anneal_epsilon = False
        self.cosine_anneal = None
        self.gamma = gamma
        self.dropout = dropout
        self.state_space = state_space
        self.hidden_size = hidden_size
        self.action_space = action_space
        
        self.train_start = min(train_start, memory_max_len)
        if train_start >= memory_max_len:
            print(f"Truncated agent.train_start to agent.memory_max_len: {memory_max_len}.")

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_max_len)
        self.memory_max_len = memory_max_len
        self.loss_history = []

        self.policy_network = self.create_network()
        self.init_weights(self.policy_network)
        self.use_target_network = use_target_network
        if use_target_network:
            self.network_sync_rate = network_sync_rate
            self.target_network = self.create_network()
            self.copy_weights(self.policy_network, self.target_network)

    def create_network(self):
        network = nn.Sequential(
            nn.Linear(self.state_space, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_space)
        )
        network.to(self.device)
        return network

    def init_weights(self, network):
        for parameter in network.parameters():
            if not parameter.requires_grad:
                continue

            # initialize bias
            if parameter.dim() == 1:
                nn.init.zeros_(parameter)

            if parameter.dim() > 1:
                # for tanh / sigmoid activation
                # nn.init.xavier_uniform_(parameter)

                # for ReLU, leaky ReLU activations
                nn.init.kaiming_uniform_(parameter, nonlinearity='relu')

    def copy_weights(self, from_network, to_network):
        to_network.load_state_dict(from_network.state_dict())
        if self.LOG_DETAILS:
            print(f"Copied weights network weights.")

    def forward(self, states, use_target_network=False):
        """
        Performs a forward pass through either the target or policy network.

        args:
            states (torch.tensor): 
                Network inputs. Must conform to shape: 
                (self.batch_size|?, self.action_space)

            use_target_network (bool): 
                Boolean specifying whether or not to use the target
                network. If False, the policy network is used. Default if False.

        returns:
            Predicted Q values (torch.tensor) in shape of 
                (self.batch_size|?, self.action_space)
        """
        if use_target_network:
            return self.target_network(states)
        else:
            return self.policy_network(states)

    def select_action(self, state, mask, indicies):
        # select action greedily with respect to state
        # or select randomly from uniform distribution
        if np.random.rand() <= self.epsilon:
            return np.random.choice(indicies)
        else:
            with torch.no_grad():
                q_values = self.forward(state.reshape((1, self.action_space)))

                # mask out invalid actions with -inf
                q_masked = torch.where(mask != 0, q_values, -1e9)

                return torch.argmax(q_masked)

    def prep_cosine_anneal(self, epsilon_min, epsilon_max, num_episodes):
        self.anneal_epsilon = True
        self.cosine_anneal = lambda episode: epsilon_min + \
            (1/2) * (epsilon_max - epsilon_min) * \
            (1 + np.cos((episode / num_episodes) * np.pi))

    def decay_epsilon(self, episode: int = None):
        if self.anneal_epsilon:
            if self.cosine_anneal is None:
                raise ValueError(
                    f"""(decay_epsilon:DeepQAgent.py) Cosine anneal has not been prepared.
                    Run 'prep_cosine_anneal' before training.""")
            if episode is None:
                raise ValueError(
                    f"""(decay_epsilon:DeepQAgent.py)
                    Must pass episode number in order to perform cosine annealing.
                    Recieved {episode}.""")
            self.epsilon = self.cosine_anneal(episode)
        else:
            # if self.epsilon > self.epsilon_min:
            #     self.epsilon *= self.epsilon_decay_rate

            # stepsize = number of iterations before until
            # learning rate returns to initial value
            stepsize = 2000
            cycle = np.floor(1 + episode / ( 2 * stepsize))
            x = np.abs(episode/stepsize - 2*cycle + 1)
            self.epsilon = self.epsilon + (self.epsilon_max - self.epsilon) * np.max(0, 1 - x)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, optimizer, criterion, episode: int = None):
        memory_length = len(self.memory)
        if memory_length < self.train_start or memory_length < self.batch_size:
            return

        batch = random.sample(self.memory, min(memory_length, self.batch_size))

        # preprocess batch
        q_batch = []
        target_q_batch = []
        for state, action, reward, next_state, done in batch:

            state = state.float().reshape((1, self.action_space))
            q_values = self.forward(state).squeeze()
            for i, cell in enumerate(state):
                if cell != 0:
                    q_values[i] = -1e9
            
            target_q_values = q_values.detach().clone()
            
            next_state = next_state.float().reshape((1, self.action_space))
            q_values_next = self.forward(
                next_state,
                use_target_network=self.use_target_network).squeeze()
            for i, cell in enumerate(next_state):
                if cell != 0:
                    q_values_next[i] = -1e9

            if done:
                target_q_values[action] = reward
            else:
                
                # Below is the Bellman equation. It stipulates that the
                # state-action (Q) value is its reward plus the max of the
                # rewards of the next state-action Q values.

                # Q(s, a) = reward + γ * Q(s', a')

                # γ is Gamma, the discount factor. This helps train the
                # model that rewards that are further away are less valuable.
                with torch.no_grad():
                    target_q_values[action] = reward + \
                        (self.gamma * torch.max(q_values_next))

            q_batch.append(q_values)
            target_q_batch.append(target_q_values)
        
        # optimize the model
        loss = criterion(torch.stack(q_batch), torch.stack(target_q_batch))
        self.loss_history.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping in-place
        total_norm = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        
        if self.LOG_DETAILS:
            print(f"\nTarget Q (sample): {target_q_batch[0]}")
            print(f"clipped gradient norm: {total_norm}")
            print(f"loss: {loss.item()}")

        optimizer.step()
        
        self.decay_epsilon(episode)

    def get_loss_history(self, items_from_back: int = 0):
        if items_from_back <= 0:
            return self.loss_history
        return self.loss_history[-items_from_back:]

    def plot_loss_history(self):
        plt.clf()
        plt.title("Average Loss per Replay")
        plt.xlabel("Replays")
        plt.ylabel("Loss")

        # plot raw loss history
        plt.plot(self.loss_history)

        # plot 100 episode averages (from pytorch example)
        if len(self.loss_history) >= 100:
            loss_tensor = torch.tensor(self.loss_history)
            means = loss_tensor.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.show()

    def clear_loss_history(self):
        del self.loss_history
        self.loss_history = []

    def save_model(self, destination_path: str = "./", name: str = ""):
        if not os.path.exists(destination_path):
            os.makedirs(destination_path, exist_ok=True)

        match = re.search(r'\d+K\+-?\d+K', name, re.IGNORECASE)
        if match:
            tag = match.group(0)
            total = eval(tag.upper().replace("K", "").replace("-", ""))
            name = name.replace(tag, f"{total}K")

        filename = name if name.endswith(
            self.PT_EXTENSION) else name + self.PT_EXTENSION
        filepath = os.path.join(destination_path, filename)

        torch.save(self.policy_network.state_dict(), filepath)
        print(f"Model saved to '{filepath}'.")

        return filepath

    def load_model(self, filepath: str = ""):
        self.policy_network.load_state_dict(torch.load(
            filepath, weights_only=True, map_location=self.device))
        print(f"Model loaded from '{filepath}'.")

        if self.use_target_network:
            self.copy_weights(self.policy_network, self.target_network)
