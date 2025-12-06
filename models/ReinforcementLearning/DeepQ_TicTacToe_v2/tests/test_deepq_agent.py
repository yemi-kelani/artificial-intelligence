"""
Unit tests for DeepQAgent class.
"""

import unittest
import torch
import numpy as np
from collections import deque
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.ReinforcementLearning.DeepQ_TicTacToe_v2.DeepQAgent import DeepQAgent


class TestDeepQAgent(unittest.TestCase):
    """Test cases for DeepQAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.agent = DeepQAgent(
            device=self.device,
            epsilon=1.0,
            gamma=0.99,
            state_space=9,
            action_space=9,
            hidden_size=128,
            dropout=0.1,
            train_start=100,
            batch_size=32,
            memory_max_len=1000
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.device, self.device)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertEqual(self.agent.state_space, 9)
        self.assertEqual(self.agent.action_space, 9)
        self.assertIsInstance(self.agent.memory, deque)
        self.assertEqual(self.agent.memory.maxlen, 1000)
    
    def test_device_consistency(self):
        """Test device consistency validation."""
        # Should not raise any errors
        self.agent._validate_device_consistency()
        
        # Check all parameters are on correct device
        for param in self.agent.policy_network.parameters():
            self.assertEqual(param.device, self.device)
        
        if self.agent.use_target_network:
            for param in self.agent.target_network.parameters():
                self.assertEqual(param.device, self.device)
    
    def test_epsilon_decay(self):
        """Test epsilon decay functionality."""
        initial_epsilon = self.agent.epsilon
        
        # Test exponential decay
        self.agent.anneal_epsilon = False
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)
        
        # Test cosine annealing
        self.agent.epsilon = 1.0
        self.agent.prep_cosine_anneal(0.01, 1.0, 1000)
        self.agent.decay_epsilon(episode=500)
        self.assertLess(self.agent.epsilon, 1.0)
        self.assertGreater(self.agent.epsilon, 0.01)
    
    def test_remember(self):
        """Test experience memory functionality."""
        state = torch.zeros(9)
        action = 4
        reward = 1.0
        next_state = torch.zeros(9)
        done = False
        
        # Test remembering experience
        initial_memory_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.memory), initial_memory_len + 1)
        
        # Test memory overflow
        for i in range(self.agent.memory_max_len + 10):
            self.agent.remember(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.memory), self.agent.memory_max_len)
    
    def test_select_action(self):
        """Test action selection."""
        state = torch.zeros(9)
        mask = torch.ones(9)
        indices = list(range(9))
        
        # Test random selection (epsilon = 1.0)
        self.agent.epsilon = 1.0
        actions = []
        for _ in range(100):
            action = self.agent.select_action(state, mask, indices)
            actions.append(action)
        
        # Should get various actions with high epsilon
        self.assertGreater(len(set(actions)), 1)
        
        # Test greedy selection (epsilon = 0.0)
        self.agent.epsilon = 0.0
        action = self.agent.select_action(state, mask, indices)
        self.assertIn(action, indices)
    
    def test_forward_pass(self):
        """Test forward pass through networks."""
        states = torch.randn(32, 9)
        
        # Test policy network forward
        q_values = self.agent.forward(states, use_target_network=False)
        self.assertEqual(q_values.shape, (32, 9))
        
        # Test target network forward
        if self.agent.use_target_network:
            q_values_target = self.agent.forward(states, use_target_network=True)
            self.assertEqual(q_values_target.shape, (32, 9))
    
    def test_mask_invalid(self):
        """Test invalid action masking."""
        # Single state
        q_values = torch.randn(9)
        state = torch.tensor([1, 0, -1, 0, 0, 0, 0, 0, 0])
        masked_q = self.agent.mask_invalid(q_values, state)
        
        # Check occupied positions are masked
        self.assertEqual(masked_q[0], -1e9)
        self.assertEqual(masked_q[2], -1e9)
        
        # Batch of states
        q_values_batch = torch.randn(2, 9)
        states_batch = torch.tensor([
            [1, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, -1, 0]
        ])
        masked_q_batch = self.agent.mask_invalid(q_values_batch, states_batch)
        
        # Check masking for each batch
        self.assertEqual(masked_q_batch[0, 0], -1e9)
        self.assertEqual(masked_q_batch[0, 2], -1e9)
        self.assertEqual(masked_q_batch[1, 4], -1e9)
        self.assertEqual(masked_q_batch[1, 7], -1e9)
    
    def test_save_load_model(self):
        """Test model saving and loading."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            filepath = self.agent.save_model(temp_dir, "test_model")
            self.assertTrue(os.path.exists(filepath))

            # Create new agent with SAME architecture (hidden_size=128 to match self.agent)
            new_agent = DeepQAgent(
                device=self.device,
                epsilon=0.5,
                gamma=0.99,
                state_space=9,
                action_space=9,
                hidden_size=128  # Match the test agent's hidden_size
            )

            # Epsilon should be different before loading
            self.assertNotEqual(new_agent.epsilon, self.agent.epsilon)

            # Load model weights
            new_agent.load_model(filepath)

            # Check weights are loaded correctly
            for (p1, p2) in zip(self.agent.policy_network.parameters(),
                               new_agent.policy_network.parameters()):
                self.assertTrue(torch.allclose(p1, p2))


class TestDeviceHandling(unittest.TestCase):
    """Test device handling and CUDA compatibility."""
    
    def test_cpu_device(self):
        """Test CPU device handling."""
        agent = DeepQAgent(
            device="cpu",
            epsilon=1.0,
            gamma=0.99,
            state_space=9,
            action_space=9
        )
        self.assertEqual(agent.device.type, "cpu")
        agent._validate_device_consistency()
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        """Test CUDA device handling."""
        agent = DeepQAgent(
            device="cuda",
            epsilon=1.0,
            gamma=0.99,
            state_space=9,
            action_space=9
        )
        self.assertEqual(agent.device.type, "cuda")
        agent._validate_device_consistency()
    
    def test_device_string_conversion(self):
        """Test device string to torch.device conversion."""
        agent = DeepQAgent(
            device="cpu",
            epsilon=1.0,
            gamma=0.99,
            state_space=9,
            action_space=9
        )
        self.assertIsInstance(agent.device, torch.device)


if __name__ == "__main__":
    unittest.main()