"""
Configuration management for DQN TicTacToe v2.
Centralizes all hyperparameters and settings.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class DQNConfig:
    """Deep Q-Network hyperparameters configuration."""
    learning_rate: float = 0.001
    momentum: float = 0.90
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_max: float = 1.0
    epsilon_decay_rate: float = 0.999
    gamma: float = 0.99
    state_space: int = 9
    action_space: int = 9
    hidden_size: int = 256
    dropout: float = 0.1
    train_start: int = 1000
    batch_size: int = 64
    memory_max_len: int = 10000
    use_target_network: bool = True
    network_sync_rate: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DQNConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Training process configuration."""
    num_episodes: int = 100000
    save_frequency: int = 5000
    test_frequency: int = 10000
    validation_episodes: int = 1000
    print_frequency: int = 10
    checkpoint_frequency: int = 100
    save_path: str = "./trained_models/ReinforcementLearning/TicTacToeV2"
    model_name_prefix: str = "TicTacToe-v2"
    random_seed: Optional[int] = 42
    device: str = "auto"  # "auto", "cuda", or "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class GameConfig:
    """Game environment configuration."""
    opponent_level: str = "naive"  # "naive", "optimal", or "agent"
    start_as_X: bool = True
    win_reward: float = 1.0
    tie_reward: float = 0.3
    loss_reward: float = -1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GameConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str = "default"
    description: str = ""
    dqn: DQNConfig = None
    training: TrainingConfig = None
    game: GameConfig = None
    
    def __post_init__(self):
        """Initialize nested configs if not provided."""
        if self.dqn is None:
            self.dqn = DQNConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.game is None:
            self.game = GameConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert complete config to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "dqn": self.dqn.to_dict(),
            "training": self.training.to_dict(),
            "game": self.game.to_dict()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        dqn_config = DQNConfig.from_dict(config_dict.get("dqn", {}))
        training_config = TrainingConfig.from_dict(config_dict.get("training", {}))
        game_config = GameConfig.from_dict(config_dict.get("game", {}))
        
        return cls(
            experiment_name=config_dict.get("experiment_name", "default"),
            description=config_dict.get("description", ""),
            dqn=dqn_config,
            training=training_config,
            game=game_config
        )
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig(
        experiment_name="default",
        description="Default DQN TicTacToe configuration"
    )


def get_config_presets() -> Dict[str, ExperimentConfig]:
    """Get predefined configuration presets."""
    presets = {}
    
    # Baseline configuration
    presets["baseline"] = ExperimentConfig(
        experiment_name="baseline",
        description="Untrained baseline model for comparison"
    )
    
    # Fast training configuration
    fast_config = get_default_config()
    fast_config.experiment_name = "fast_training"
    fast_config.description = "Quick training for testing"
    fast_config.training.num_episodes = 10000
    fast_config.training.save_frequency = 1000
    fast_config.dqn.memory_max_len = 5000
    presets["fast"] = fast_config
    
    # Optimal opponent configuration
    optimal_config = get_default_config()
    optimal_config.experiment_name = "optimal_opponent"
    optimal_config.description = "Training against optimal minimax opponent"
    optimal_config.game.opponent_level = "optimal"
    optimal_config.dqn.epsilon = 0.5  # Lower initial exploration
    presets["optimal"] = optimal_config
    
    # Self-play configuration
    selfplay_config = get_default_config()
    selfplay_config.experiment_name = "self_play"
    selfplay_config.description = "Agent playing against itself"
    selfplay_config.game.opponent_level = "agent"
    selfplay_config.dqn.epsilon_decay_rate = 0.9995  # Slower decay for self-play
    presets["selfplay"] = selfplay_config
    
    # Large model configuration
    large_config = get_default_config()
    large_config.experiment_name = "large_model"
    large_config.description = "Larger network architecture"
    large_config.dqn.hidden_size = 512
    large_config.dqn.batch_size = 128
    large_config.dqn.memory_max_len = 20000
    presets["large"] = large_config
    
    return presets


def create_config_from_args(args: Dict[str, Any]) -> ExperimentConfig:
    """
    Create configuration from command-line arguments or dictionary.
    
    Args:
        args: Dictionary of configuration parameters
        
    Returns:
        ExperimentConfig instance
    """
    # Start with default config
    config = get_default_config()
    
    # Check for preset
    if "preset" in args:
        presets = get_config_presets()
        if args["preset"] in presets:
            config = presets[args["preset"]]
    
    # Override with specific arguments
    for key, value in args.items():
        if key == "preset":
            continue
        
        # Parse nested keys (e.g., "dqn.learning_rate")
        parts = key.split(".")
        if len(parts) == 2:
            section, param = parts
            if hasattr(config, section):
                section_config = getattr(config, section)
                if hasattr(section_config, param):
                    setattr(section_config, param, value)
    
    return config