import os
import re
import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from .logger import get_logger


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
        memory_max_len: int = 10000,
        use_target_network: bool = True,
        network_sync_rate: int = 10
    ):

        super(DeepQAgent, self).__init__()
        self.PT_EXTENSION = ".pt"
        self.LOG_DETAILS = False
        self.logger = get_logger(self.__class__.__name__)

        # Validate and set device
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
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
            self.logger.warning(f"Truncated agent.train_start to agent.memory_max_len: {memory_max_len}.")

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_max_len)
        self.memory_max_len = memory_max_len
        self.loss_history = []  # Bounded loss history

        self.policy_network = self.create_network()
        self.init_weights(self.policy_network)
        self.use_target_network = use_target_network
        if use_target_network:
            self.network_sync_rate = network_sync_rate
            self.target_network = self.create_network()
            self.copy_weights(self.policy_network, self.target_network)
        
        # Validate device consistency
        self._validate_device_consistency()

    def create_network(self):
        network = nn.Sequential(
            nn.Linear(self.state_space, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
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
                # Use kaiming or ReLU / leaky ReLU activations
                # nn.init.kaiming_uniform_(parameter, nonlinearity='relu')
                nn.init.kaiming_normal_(parameter, nonlinearity="relu")

    def copy_weights(self, from_network, to_network):
        to_network.load_state_dict(from_network.state_dict())
        if self.LOG_DETAILS:
            self.logger.debug(f"Copied weights network weights.")
    
    def _validate_device_consistency(self):
        """Validate that all model components are on the same device."""
        # Check policy network
        policy_device = next(self.policy_network.parameters()).device
        if policy_device != self.device:
            raise RuntimeError(f"Policy network is on {policy_device} but agent device is {self.device}")
        
        # Check target network if it exists
        if self.use_target_network:
            target_device = next(self.target_network.parameters()).device
            if target_device != self.device:
                raise RuntimeError(f"Target network is on {target_device} but agent device is {self.device}")
        
        if self.LOG_DETAILS:
            self.logger.debug(f"✓ Device consistency validated: all components on {self.device}")

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

    def select_action(self, state, mask, indices):
        """Select action using epsilon-greedy policy.

        Args:
            state: Current game state tensor
            mask: Valid action mask (1 for valid, 0 for invalid)
            indices: List of valid action indices

        Returns:
            int: Selected action index
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(indices)
        else:
            with torch.no_grad():
                # Ensure state is on correct device
                state = state.to(self.device)
                mask = mask.to(self.device)
                q_values = self.forward(state.reshape((1, self.state_space)))

                # mask out invalid actions with -inf
                q_masked = torch.where(mask != 0, q_values, -1e9)

                # handle equal Q-values by adding small noise
                noise = torch.randn_like(q_masked) * 1e-6
                q_masked = q_masked + noise

                return torch.argmax(q_masked).item()

    def prep_cosine_anneal(self, epsilon_min, epsilon_max, num_episodes):
        self.anneal_epsilon = True
        self.cosine_anneal = lambda episode: epsilon_min + \
            (1/2) * (epsilon_max - epsilon_min) * \
            (1 + np.cos((episode / num_episodes) * np.pi))

    def decay_epsilon(self, episode: int = -1):
        if self.anneal_epsilon:
            if self.cosine_anneal is None:
                raise ValueError(
                    f"""(decay_epsilon:DeepQAgent.py) Cosine anneal has not been prepared.
                    Run 'prep_cosine_anneal' before training.""")
            if episode < 0:
                raise ValueError(
                    f"""(decay_epsilon:DeepQAgent.py)
                    Must pass episode number in order to perform cosine annealing.
                    Received {episode}.""")
            self.epsilon = self.cosine_anneal(episode)
        elif not self.anneal_epsilon:  # Only do exponential decay if not using cosine
            # Exponential decay
            self.epsilon = self.epsilon * self.epsilon_decay_rate
        
        # Ensure epsilon stays within bounds
        self.epsilon = np.clip(self.epsilon, self.epsilon_min, self.epsilon_max)

    def remember(self, state, action, reward, next_state, done):
        # Ensure states are on the correct device before storing
        state = state.to(self.device) if torch.is_tensor(state) else state
        next_state = next_state.to(self.device) if torch.is_tensor(next_state) else next_state
        self.memory.append((state, action, reward, next_state, done))
    
    def mask_invalid(self, q_values, state):
        """Mask invalid actions in Q-values based on game state.
        
        Args:
            q_values: Q-values tensor, shape (batch_size, action_space) or (action_space,)
            state: Game state tensor, shape (batch_size, state_space) or (state_space,)
        """
        # Handle both batch and single input
        if q_values.dim() == 1:
            q_values_masked = q_values.clone().detach().to(self.device)
            state = state.to(self.device)
            for i, cell in enumerate(state):
                if cell != 0:
                    q_values_masked[i] = -1e9
        else:
            # Batch processing
            q_values_masked = q_values.clone().detach().to(self.device)
            state = state.to(self.device)
            for batch_idx in range(q_values.size(0)):
                for i, cell in enumerate(state[batch_idx]):
                    if cell != 0:
                        q_values_masked[batch_idx, i] = -1e9
        
        return q_values_masked

    def replay(self, optimizer, criterion, episode: int = -1):
        memory_length = len(self.memory)
        if memory_length < self.train_start or memory_length < self.batch_size:
            return

        # Thread-safe memory sampling with validation
        try:
            current_memory_length = len(self.memory)
            if current_memory_length < self.batch_size:
                return
            batch = random.sample(list(self.memory), min(current_memory_length, self.batch_size))
        except (IndexError, ValueError):
            return  # Skip this replay if memory is being modified

        q_batch = []
        target_q_batch = []
        for state, action, reward, next_state, done in batch:
            # Ensure all tensors are on correct device
            state = state.float().reshape(-1).to(self.device)
            next_state = next_state.float().reshape(-1).to(self.device)
            
            q_values = self.forward(state).squeeze()
            target_q_values = q_values.detach().clone()
            # reward = (reward - rewards_mean) / rewards_std
            
            
            if self.use_target_network:
                # Double DQN
                with torch.no_grad():
                    # Use policy network to select action
                    q_values_next_policy = self.forward(next_state, use_target_network=False).squeeze()
                    # Use target network to evaluate the action
                    q_values_next_target = self.forward(next_state, use_target_network=True).squeeze()
                    # Mask invalid actions for policy network
                    q_masked_policy = self.mask_invalid(q_values_next_policy, next_state)
                    # Select best action using policy network
                    best_action = torch.argmax(q_masked_policy)
                    # Evaluate that action using target network
                    if not done:
                        target_q_values[action] = reward + self.gamma * q_values_next_target[best_action]
                    else:
                        target_q_values[action] = reward
            else:
                # Standard DQN
                with torch.no_grad():
                #   Below is the Bellman equation. It stipulates that the
                #   state-action (Q) value is its reward plus the max of the
                #   rewards of the next state-action Q values.

                #   Q(s, a) = reward + γ * Q(s', a')

                #   γ is Gamma, the discount factor. This helps train the
                #   model that rewards that are further away are less valuable.
                    if not done:
                        q_values_next = self.forward(next_state, use_target_network=False).squeeze()
                        q_masked_next = self.mask_invalid(q_values_next, next_state)
                        target_q_values[action] = reward + self.gamma * torch.max(q_masked_next)
                    else:
                        target_q_values[action] = reward
            
            q_batch.append(q_values)
            target_q_batch.append(target_q_values)

        loss = criterion(torch.stack(q_batch), torch.stack(target_q_batch))
        self.loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping
        total_norm = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        
        if self.LOG_DETAILS:
            self.logger.debug(f"Target Q (sample): {target_q_batch[0]}")
            self.logger.debug(f"clipped gradient norm: {total_norm}")
            self.logger.debug(f"loss: {loss.item()}")

        optimizer.step()

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
        
    def save_checkpoint(self, destination_path: str = "./", filename: str = "DQN_checkpoint.pt"):
        torch.save({
            "policy_state_dict": self.policy_network.state_dict(),
            "target_state_dict": self.target_network.state_dict() if self.use_target_network else None,
            "epsilon": self.epsilon,
            "memory": list(self.memory),
            "loss_history": self.loss_history
        }, os.path.join(destination_path, filename))

    def save_model(self, destination_path: str = "./", name: str = ""):
        if not os.path.exists(destination_path):
            os.makedirs(destination_path, exist_ok=True)

        # Handle compound episode tags like "50K+50K" -> "100K"
        match = re.search(r'(\d+)K\+(\d+)K', name, re.IGNORECASE)
        if match:
            tag = match.group(0)
            total = int(match.group(1)) + int(match.group(2))
            name = name.replace(tag, f"{total}K")

        filename = name if name.endswith(
            self.PT_EXTENSION) else name + self.PT_EXTENSION
        filepath = os.path.join(destination_path, filename)

        torch.save(self.policy_network.state_dict(), filepath)
        self.logger.info(f"Model saved to '{filepath}'.")

        return filepath

    def load_model(self, filepath: str = "", weights_only: bool = True):
        self.policy_network.load_state_dict(torch.load(
            filepath, weights_only=weights_only, map_location=self.device))
        self.logger.info(f"Model loaded from '{filepath}'.")

        if self.use_target_network:
            self.copy_weights(self.policy_network, self.target_network)
        
        # Validate device consistency after loading
        self._validate_device_consistency()
