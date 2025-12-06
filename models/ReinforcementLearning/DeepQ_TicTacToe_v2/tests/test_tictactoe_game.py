"""
Unit tests for TicTacToeGame class.
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.ReinforcementLearning.DeepQ_TicTacToe_v2.TicTacToeGame import TicTacToeGame, OPPONENT_LEVEL
from models.ReinforcementLearning.DeepQ_TicTacToe_v2.DeepQAgent import DeepQAgent


class TestTicTacToeGame(unittest.TestCase):
    """Test cases for TicTacToeGame class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.agent = DeepQAgent(
            device=self.device,
            epsilon=0.1,
            gamma=0.99,
            state_space=9,
            action_space=9
        )
        self.game = TicTacToeGame(
            self.device, 
            self.agent, 
            OPPONENT_LEVEL.NAIVE, 
            start_as_X=True
        )
    
    def test_initialization(self):
        """Test game initialization."""
        self.assertEqual(self.game.device, self.device)
        self.assertEqual(self.game.opponent_level, OPPONENT_LEVEL.NAIVE)
        self.assertEqual(self.game.board.shape, (3, 3))
        self.assertTrue(torch.all(self.game.board == 0).item() or 
                       torch.sum(torch.abs(self.game.board)) == 1)  # Empty or one move
    
    def test_role_assignment(self):
        """Test role assignment."""
        # Test starting as X (opponent)
        game_x = TicTacToeGame(self.device, None, OPPONENT_LEVEL.NAIVE, start_as_X=True)
        self.assertEqual(game_x.role, game_x.X)
        self.assertEqual(game_x.agent_role, game_x.O)
        
        # Test starting as O (opponent)
        game_o = TicTacToeGame(self.device, None, OPPONENT_LEVEL.NAIVE, start_as_X=False)
        self.assertEqual(game_o.role, game_o.O)
        self.assertEqual(game_o.agent_role, game_o.X)
    
    def test_action_position_conversion(self):
        """Test conversion between action and position."""
        test_cases = [
            (0, (0, 0)), (1, (0, 1)), (2, (0, 2)),
            (3, (1, 0)), (4, (1, 1)), (5, (1, 2)),
            (6, (2, 0)), (7, (2, 1)), (8, (2, 2))
        ]
        
        for action, expected_pos in test_cases:
            pos = self.game.convert_action_to_position(action)
            self.assertEqual(pos, expected_pos)
            
            reconstructed_action = self.game.convert_position_to_action(*pos)
            self.assertEqual(reconstructed_action, action)
    
    def test_valid_moves(self):
        """Test valid move detection."""
        # Empty board should have all moves valid
        self.game.board = torch.zeros((3, 3))
        valid_moves, mask, indices = self.game.get_valid_moves()
        self.assertEqual(len(valid_moves), 9)
        self.assertEqual(torch.sum(mask).item(), 9)
        self.assertEqual(len(indices), 9)
        
        # Board with some moves should have fewer valid moves
        self.game.board[0, 0] = self.game.X
        self.game.board[1, 1] = self.game.O
        valid_moves, mask, indices = self.game.get_valid_moves()
        self.assertEqual(len(valid_moves), 7)
        self.assertEqual(torch.sum(mask).item(), 7)
        self.assertNotIn(0, indices)
        self.assertNotIn(4, indices)
    
    def test_is_valid_move(self):
        """Test move validation."""
        self.game.board = torch.zeros((3, 3))
        
        # Valid moves on empty board
        self.assertTrue(self.game.is_valid_move(0, 0, self.game.X))
        self.assertTrue(self.game.is_valid_move(2, 2, self.game.O))
        
        # Invalid moves - out of bounds
        self.assertFalse(self.game.is_valid_move(-1, 0, self.game.X))
        self.assertFalse(self.game.is_valid_move(3, 0, self.game.X))
        self.assertFalse(self.game.is_valid_move(0, -1, self.game.X))
        self.assertFalse(self.game.is_valid_move(0, 3, self.game.X))
        
        # Invalid moves - occupied cell
        self.game.board[1, 1] = self.game.X
        self.assertFalse(self.game.is_valid_move(1, 1, self.game.O))
    
    def test_game_over_detection(self):
        """Test game over detection."""
        # Test horizontal win
        self.game.board = torch.tensor([
            [1, 1, 1],
            [0, -1, 0],
            [-1, 0, 0]
        ]).float()
        reward, done, winner = self.game.is_game_over()
        self.assertTrue(done)
        self.assertEqual(winner, 1)
        
        # Test vertical win
        self.game.board = torch.tensor([
            [-1, 1, 0],
            [-1, 1, 0],
            [-1, 0, 0]
        ]).float()
        reward, done, winner = self.game.is_game_over()
        self.assertTrue(done)
        self.assertEqual(winner, -1)
        
        # Test diagonal win
        self.game.board = torch.tensor([
            [1, -1, 0],
            [-1, 1, 0],
            [0, 0, 1]
        ]).float()
        reward, done, winner = self.game.is_game_over()
        self.assertTrue(done)
        self.assertEqual(winner, 1)
        
        # Test tie
        self.game.board = torch.tensor([
            [1, -1, 1],
            [-1, 1, -1],
            [-1, 1, -1]
        ]).float()
        reward, done, winner = self.game.is_game_over()
        self.assertTrue(done)
        self.assertIsNone(winner)
        
        # Test ongoing game
        self.game.board = torch.tensor([
            [1, -1, 0],
            [0, 1, 0],
            [0, 0, -1]
        ]).float()
        reward, done, winner = self.game.is_game_over()
        self.assertFalse(done)
        self.assertIsNone(winner)
    
    def test_reward_calculation(self):
        """Test reward calculation for different game states."""
        # Set agent role
        self.game.agent_role = self.game.O
        
        # Test win reward
        self.game.board = torch.tensor([
            [1, 1, 1],
            [-1, -1, 0],
            [0, 0, 0]
        ]).float()
        reward, _, _ = self.game.is_game_over()
        self.assertEqual(reward, self.game.WIN_REWARD)
        
        # Test loss reward
        self.game.agent_role = self.game.X
        reward, _, _ = self.game.is_game_over()
        self.assertEqual(reward, self.game.LOSS_REWARD)
        
        # Test tie reward
        self.game.board = torch.tensor([
            [1, -1, 1],
            [-1, 1, -1],
            [-1, 1, -1]
        ]).float()
        reward, _, _ = self.game.is_game_over()
        self.assertEqual(reward, self.game.TIE_REWARD)
    
    def test_flip_roles(self):
        """Test role flipping."""
        initial_role = self.game.role
        initial_agent_role = self.game.agent_role
        
        self.game.flip_roles()
        
        self.assertEqual(self.game.role, -initial_role)
        self.assertEqual(self.game.agent_role, initial_role)
    
    def test_reset(self):
        """Test game reset."""
        # Make some moves
        self.game.board[0, 0] = self.game.X
        self.game.board[1, 1] = self.game.O
        
        # Reset without flipping roles
        self.game.reset(flip_roles=False)
        
        # Board should be empty or have one move (if X starts)
        total_moves = torch.sum(torch.abs(self.game.board)).item()
        self.assertIn(total_moves, [0, 1])
        
        # Reset with flipping roles
        initial_role = self.game.role
        self.game.reset(flip_roles=True)
        self.assertEqual(self.game.role, -initial_role)
    
    def test_opponent_levels(self):
        """Test different opponent levels."""
        # Test setting valid opponent levels
        for level in [OPPONENT_LEVEL.NAIVE, OPPONENT_LEVEL.OPTIMAL, OPPONENT_LEVEL.AGENT]:
            self.game.set_opponent_level(level)
            self.assertEqual(self.game.opponent_level, level)
        
        # Test invalid opponent level
        with self.assertRaises(Exception):
            self.game.set_opponent_level("invalid_level")


class TestMinimaxAlgorithm(unittest.TestCase):
    """Test minimax algorithm implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.game = TicTacToeGame(self.device, None, OPPONENT_LEVEL.OPTIMAL)
    
    def test_minimax_win_detection(self):
        """Test minimax detects winning moves."""
        # Set up a board where X can win in one move
        self.game.board = torch.tensor([
            [-1, -1, 0],  # X can win by playing (0, 2)
            [1, 1, 0],    # O needs to block at (1, 2)
            [0, 0, 0]
        ]).float()
        self.game.role = self.game.X
        
        # Minimax should choose the winning move
        board = self.game.board.clone()
        score = self.game.minmax(self.game.X, self.game.X, board)
        self.assertEqual(score, 1)  # X wins
    
    def test_minimax_blocking(self):
        """Test minimax prioritizes winning over blocking."""
        # Create a fresh OPTIMAL game where X is the opponent (environment role)
        self.game = TicTacToeGame(self.device, None, OPPONENT_LEVEL.OPTIMAL, start_as_X=True)
        self.game.reset_board()

        # Board setup where X (the opponent) can win at (0, 2)
        # X has 2 pieces, O has 2 pieces, 4 total moves = remainder 0 = X's turn
        self.game.board = torch.tensor([
            [-1, -1, 0],  # X at (0,0), (0,1) - X can win at (0,2)
            [1, 1, 0],    # O at (1,0), (1,1) - O could win at (1,2) if it were O's turn
            [0, 0, 0]
        ]).float()
        self.game.role = self.game.X  # Opponent plays as X

        # X should take the winning move at (0, 2)
        self.game.move()

        # Check that X played at (0, 2) to win
        self.assertEqual(self.game.board[0, 2].item(), self.game.X)


if __name__ == "__main__":
    unittest.main()