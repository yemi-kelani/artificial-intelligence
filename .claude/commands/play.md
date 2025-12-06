# Play TicTacToe

Play an interactive TicTacToe game against a trained AI agent.

## Usage
```bash
# Play against baseline model
python -m models.ReinforcementLearning.DeepQ_TicTacToe_v2.play TicTacToe-v2-BASELINE

# Play against NAIVE trained model (specify K iterations)
python -m models.ReinforcementLearning.DeepQ_TicTacToe_v2.play TicTacToe-v2-NAIVE-10K

# Play against OPTIMAL trained model
python -m models.ReinforcementLearning.DeepQ_TicTacToe_v2.play TicTacToe-v2-OPTIMAL-10K
```

## Available Models
- `TicTacToe-v2-BASELINE` - Untrained baseline
- `TicTacToe-v2-NAIVE-*K` - Trained against random opponents
- `TicTacToe-v2-OPTIMAL-*K` - Trained against optimal minimax opponents