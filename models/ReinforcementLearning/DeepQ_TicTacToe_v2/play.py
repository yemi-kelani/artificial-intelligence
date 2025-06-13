import re
import torch
import argparse
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
        device             = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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

    agent.load_model(filepath=MODEL_PATH)

    return agent


def construct_prompt(text: str):
    return "\n" + colors.wrap_text(text, colors.YELLOW) + "\n"


def get_player_action():
    response = input(construct_prompt("Your turn..."))
    
    if response.lower() == "q":
        exit()

    single_input_re = r'^\s*[1-9]\s*$'
    two_input_re1 = r'^\s*[1-3],\s*[1-3]\s*$'
    two_input_re2 = r'^\s*[1-3]\s+[1-3]\s*$'

    if re.match(single_input_re, response):
        return eval(response.strip()) - 1

    if re.match(two_input_re1, response):
        digits = response.strip().replace(' ', '').split(',')
        row = eval(digits[0]) - 1
        col = eval(digits[-1]) - 1
        return ((3 * row) + col)

    if re.match(two_input_re2, response):
        digits = response.strip().split(' ')
        row = eval(digits[0]) - 1
        col = eval(digits[-1]) - 1
        return ((3 * row) + col)
    
    print(
        colors.wrap_text(
            f"""
            Invalid input: {response}.
            Regex used to check input: {single_input_re}, {two_input_re1},
            {two_input_re2}
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
            colors.CYAN
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
        _, done, _ = environment.move()
        
        environment.print_state()
        print()

        if done:
            game_stats["games"] += 1
            if reward == environment.WIN_REWARD:
                game_stats["wins"] += 1
            elif reward == environment.LOSS_REWARD:
                game_stats["losses"] += 1
            elif reward == environment.TIE_REWARD:
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

    parser.add_argument('model_identifier')
    args = parser.parse_args()

    if type(args.model_identifier) is not str:
        raise ValueError(
            f"""
              Error: Second argument to must be a valid string specifiying a model name.
              Models are located in 'trained_models/ReinforcementLearning/TicTacToeV2'.
              Recieved {args.model_identifier} instead.
            """
        )

    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = determine_model_name(args.model_identifier)
    except:
        raise ValueError(
            f"""
              Error: Second argument to must be a valid string specifiying a model name.
              Models are located in 'trained_models/ReinforcementLearning/TicTacToeV2'.
              Recieved {args.model_identifier} instead.
            """
        )

    enemy = supply_model(model_name, DEVICE)
    environment = TicTacToeGame(
        DEVICE, enemy, OPPONENT_LEVEL.AGENT, start_as_X=True)
    
    run_session(environment)
    

if __name__ == "__main__":
    play()
