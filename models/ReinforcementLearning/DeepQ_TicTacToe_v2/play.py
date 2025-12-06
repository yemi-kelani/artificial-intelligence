import re
import torch
import argparse
import os
from pathlib import Path
from models.common import colors, get_root_directory
from models.ReinforcementLearning.DeepQ_TicTacToe_v2.DeepQAgent import DeepQAgent
from models.ReinforcementLearning.DeepQ_TicTacToe_v2.TicTacToeGame import TicTacToeGame, OPPONENT_LEVEL


def supply_model(agent_name: str, device: torch.device):

    EPSILON = 1.0
    GAMMA = 0.99
    STATE_SPACE = 9
    ACTION_SPACE = 9
    HIDDEN_SIZE = 256
    DROPOUT = 0.1
    TRAIN_START = 1000
    BATCH_SIZE = 64
    MEMORY_MAX_LEN = 10000
    USE_TARGET_NETWORK = True
    NETWORK_SYNC_RATE = 100 
    
    root_directory = get_root_directory()
    MODEL_PATH = f"{root_directory}/trained_models/ReinforcementLearning/TicTacToeV2/{agent_name}.pt"

    agent = DeepQAgent(
        device             = device,  # Use the passed device parameter
        epsilon            = EPSILON,
        gamma              = GAMMA,
        state_space        = STATE_SPACE,
        action_space       = ACTION_SPACE,
        hidden_size        = HIDDEN_SIZE,
        dropout            = DROPOUT,
        train_start        = TRAIN_START,
        batch_size         = BATCH_SIZE,
        memory_max_len     = MEMORY_MAX_LEN,
        use_target_network = USE_TARGET_NETWORK,
        network_sync_rate  = NETWORK_SYNC_RATE
    )
    
    try:
        agent.load_model(filepath=MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

    return agent


def construct_prompt(text: str):
    return "\n" + colors.wrap_text(text, colors.YELLOW) + "\n"


def safe_int_parse(s: str) -> int:
    """Safely parse integer from string without eval()."""
    try:
        return int(s.strip())
    except ValueError:
        raise ValueError(f"Invalid integer: {s}")

def get_player_action():
    response = input(construct_prompt("Your turn..."))
    
    if response.lower() == "q":
        exit()

    single_input_re = r'^\s*[1-9]\s*$'
    two_input_re1 = r'^\s*[1-3],\s*[1-3]\s*$'
    two_input_re2 = r'^\s*[1-3]\s+[1-3]\s*$'

    try:
        if re.match(single_input_re, response):
            action = safe_int_parse(response) - 1
            if 0 <= action <= 8:
                return action
            else:
                raise ValueError("Action out of range")

        if re.match(two_input_re1, response):
            digits = response.strip().replace(' ', '').split(',')
            row = safe_int_parse(digits[0]) - 1
            col = safe_int_parse(digits[-1]) - 1
            if 0 <= row <= 2 and 0 <= col <= 2:
                return ((3 * row) + col)
            else:
                raise ValueError("Row/column out of range")

        if re.match(two_input_re2, response):
            digits = response.strip().split()
            row = safe_int_parse(digits[0]) - 1
            col = safe_int_parse(digits[-1]) - 1
            if 0 <= row <= 2 and 0 <= col <= 2:
                return ((3 * row) + col)
            else:
                raise ValueError("Row/column out of range")
    except (ValueError, IndexError) as e:
        print(
            colors.wrap_text(
                f"Invalid input: {response}. Error: {e}",
                colors.RED
            ))
        return -1
    
    print(
        colors.wrap_text(
            f"""
            Invalid input format: {response}.
            Expected: single digit 1-9, or row,col format like "2,3" or "2 3"
            """,
            colors.RED
        ))
    return -1


def reset():
    response = input(construct_prompt(
        "Start a new game? Enter 'y' to continue. Enter any other key to quit"))

    continue_regex = r'^\s*[yY]?\s*$'
    if re.match(continue_regex, response):
        return True
    return False


def print_game_stats(game_stats):
    num_wins = game_stats["wins"]
    num_losses = game_stats["losses"]
    num_draws = game_stats["draws"]
    num_games = game_stats["games"]
    
    print("\n******************************************************")
    print(
        colors.wrap_text(
            "Games won:  "
            + f"    {num_wins} ({round((num_wins/num_games) * 100, 4)} %)",
            colors.GREEN
        ))
    print(
        colors.wrap_text(
            "Games drawn:"
            + f"    {num_draws} ({round((num_draws/num_games) * 100, 4)}%)",
            colors.YELLOW
        ))
    print(
        colors.wrap_text(
            "Games lost: "
            + f"    {num_losses} ({round((num_losses/num_games) * 100, 4)}%)",
            colors.RED
        ))
    print("******************************************************\n")


def run_session(environment: TicTacToeGame):

    environment.agent.eval()

    game_stats = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "games": 0
    }

    print(
        """
        Please take an action via the command line when prompted. Actions can take two forms.

        1. An action can be an integer in the range of [1,9] inclusive. Each integer
        corresponds to a cell on the TicTacToe board from left to right, top to bottom.
        For instance, integers 1-3 correspond to the cells in first row of the board.

        Example input: 1
        Example input: 9
        Example input: 7

        2. Pass in two integers in the range of [1,3] inclusive, separated by a comma, and/or space.
        The first input corresponds to the row and the second input corresponds to the column.

        Example input: 1 2
        Example input: 2,3
        Example input: 3, 3
        """
    )
    environment.print_state()
    print()

    done = False
    while not done:

        _, _, valid_actions = environment.get_valid_moves()

        action = -1
        while action == -1:
            action = get_player_action()
            if action != -1 and action not in valid_actions:
                row, col = environment.convert_action_to_position(action)
                print(
                    colors.wrap_text(
                        f"""
                        Invalid input: row: {row + 1}, column: {col + 1}.
                        That cell is already filled. Select an empty cell.
                        """,
                        colors.RED
                    ))
                action = -1

        _, reward, done = environment.take_action(action)
        environment.move()
        
        _, done, winner = environment.is_game_over()
        
        environment.print_state()
        print()

        if done:
            game_stats["games"] += 1
            if winner == environment.agent_role:
                game_stats["wins"] += 1
            elif winner == environment.role:
                game_stats["losses"] += 1
            else:
                game_stats["draws"] += 1
            print_game_stats(game_stats)

            if reset():
                environment.reset()
                print(
                    colors.wrap_text(
                        "\nNew Game...\n",
                        colors.BLUE
                    ))
                environment.print_state()
                print()
                done = False


def get_available_models():
    """Dynamically search for available trained models."""
    root_directory = get_root_directory()
    models_dir = Path(root_directory) / "trained_models" / "ReinforcementLearning" / "TicTacToeV2"
    
    if not models_dir.exists():
        return []
    
    models = []
    for file_path in models_dir.glob("*.pt"):
        model_name = file_path.stem
        models.append(model_name)
    
    return sorted(models)


def print_available_models():
    """Print all available models in a formatted way."""
    models = get_available_models()
    
    if not models:
        print(colors.wrap_text("No trained models found in trained_models/ReinforcementLearning/TicTacToeV2/", colors.RED))
        return
    
    print(colors.wrap_text("\nAvailable Models:", colors.CYAN))
    print("=" * 50)
    
    for i, model in enumerate(models, 1):
        print(f"{i:2d}. {model}")
        
        # Add description for known models
        if "BASELINE" in model.upper():
            print(f"    {colors.wrap_text('(Untrained baseline for comparison)', colors.YELLOW)}")
        elif "NAIVE" in model.upper():
            print(f"    {colors.wrap_text('(Trained against random opponent)', colors.GREEN)}")
        elif "OPTIMAL" in model.upper():
            print(f"    {colors.wrap_text('(Trained against minimax opponent)', colors.BLUE)}")
        elif "FINAL" in model.upper():
            print(f"    {colors.wrap_text('(Best performing model)', colors.CYAN)}")
    
    print("=" * 50)
    print(f"\nUsage: python -m models.ReinforcementLearning.DeepQ_TicTacToe_v2.play <MODEL_NAME>")
    print(f"Example: python -m models.ReinforcementLearning.DeepQ_TicTacToe_v2.play {models[0] if models else 'MODEL_NAME'}")


def determine_model_name(model_identifier: str):
    model_id = model_identifier.strip().replace(".pt", "")
    known_models = {
        "BASELINE":  "TicTacToe-v2-BASELINE"
    }

    model_id_upper = model_id.upper()
    for _, (key, value) in enumerate(known_models.items()):
        if model_id_upper == key.upper() \
                or model_id_upper == value.upper():
            return value

    return model_id


def play():
    parser = argparse.ArgumentParser(
        prog='play',
        description="""
            Play a game of TicTacToe against a specified model via the command line.
        """,
        epilog=""
    )

    parser.add_argument('model_identifier', nargs='?', help='Name of the model to play against')
    parser.add_argument('--list', '-l', action='store_true', 
                       help='List all available trained models')
    args = parser.parse_args()

    # Handle --list flag
    if args.list:
        print_available_models()
        return
    
    # Check if model_identifier was provided
    if not args.model_identifier:
        print(colors.wrap_text("Error: Model identifier is required", colors.RED))
        print("Use --list to see available models, or provide a model name:")
        print_available_models()
        return

    if not isinstance(args.model_identifier, str):
        raise ValueError(
            f"""
              Error: Model identifier must be a valid string specifying a model name.
              Models are located in 'trained_models/ReinforcementLearning/TicTacToeV2'.
              Received {args.model_identifier} ({type(args.model_identifier)}) instead.
            """
        )

    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {DEVICE}")
        
        model_name = determine_model_name(args.model_identifier)
        print(f"Loading model: {model_name}")
        
        enemy = supply_model(model_name, DEVICE)
        environment = TicTacToeGame(
            DEVICE, enemy, OPPONENT_LEVEL.AGENT, start_as_X=True)
        
        run_session(environment)
        
    except FileNotFoundError as e:
        print(colors.wrap_text(
            f"Model file not found: {e}", colors.RED))
        print("Available models should be in 'trained_models/ReinforcementLearning/TicTacToeV2/'")
    except RuntimeError as e:
        print(colors.wrap_text(
            f"Runtime error: {e}", colors.RED))
    except Exception as e:
        print(colors.wrap_text(
            f"Unexpected error: {e}", colors.RED))
        raise
    

if __name__ == "__main__":
    play()
