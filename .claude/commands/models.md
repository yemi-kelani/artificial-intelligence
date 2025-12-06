# Model Management

Commands for managing trained models and evaluating performance.

## List Available Models
```bash
# List all trained TicTacToe models
find trained_models/ReinforcementLearning/TicTacToeV2/ -name "*.pt" -type f | sort

# Show model file sizes
ls -lh trained_models/ReinforcementLearning/TicTacToeV2/*.pt
```

## Model Information
```bash
# Check model file details
python -c "
import torch
model_path = 'trained_models/ReinforcementLearning/TicTacToeV2/TicTacToe-v2-BASELINE.pt'
checkpoint = torch.load(model_path, map_location='cpu')
print(f'Model keys: {list(checkpoint.keys())}')
if 'model_state_dict' in checkpoint:
    print(f'Model architecture: {checkpoint.get(\"model_info\", \"No info available\")}')
    print(f'Training episode: {checkpoint.get(\"episode\", \"Unknown\")}')
"
```

## Quick Performance Test
```bash
# Test model loading and basic functionality
python -c "
from models.ReinforcementLearning.DeepQ_TicTacToe_v2.DeepQAgent import DeepQAgent
from models.ReinforcementLearning.DeepQ_TicTacToe_v2.TicTacToeGame import TicTacToeGame
import torch

# Test model loading
agent = DeepQAgent()
game = TicTacToeGame()
print('✓ Model components loaded successfully')

# Test basic prediction
state = game.get_state()
action = agent.choose_action(state, epsilon=0)  # Greedy action
print(f'✓ Model prediction working: action {action}')
"
```