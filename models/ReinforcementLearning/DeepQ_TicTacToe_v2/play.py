import os
import re
import torch
import argparse
from .TicTacToeGame import TicTacToeGame, OPPONENT_LEVEL
from .DeepQAgent import DeepQAgent
from Utils import colors


def supply_model(agent_name: str, device: torch.device):

    BATCH_SIZE = 128
    STATE_SPACE = 9
    ACTION_SPACE = 9
    EPSILON = 1.0
    GAMMA = 0.99
    HIDDEN_SIZE = 128
    EPSILON = 1.0
    GAMMA = 0.99
    DROPOUT = 0.25
    TRAIN_START = 1500
    NEGATIVE_SLOPE = 0.01
    MODEL_PATH = f"../../trained_models/ReinforcementLearning/TicTacToeV2/{
        agent_name}.pt"

    agent = DeepQAgent(
        device=DEVICE,
        epsilon=EPSILON,
        gamma=GAMMA,
        state_space=STATE_SPACE,
        action_space=ACTION_SPACE,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        train_start=TRAIN_START,
        batch_size=BATCH_SIZE,
        negative_slope=NEGATIVE_SLOPE
    )

    agent.load_model(filepath=MODEL_PATH)

    return agent


def construct_prompt(text: str):
    return "\n" + colors.wrap_text(text, colors.YELLOW) + "\n"


def get_player_action():
    response = input(construct_prompt("Your turn..."))

    single_input_re = r'^\s*[1-9]\s*$'
    two_input_re = r'^\s*[1-3],?\s+[1-3]\s*$'

    if re.match(single_input_re, response):
        return eval(response.strip()) - 1

    if re.match(two_input_re, response):
        digits = response.strip().replace(',', '').split(" ")
        row = eval(digits[0]) - 1
        col = eval(digits[-1]) - 1
        return ((3 * row) + col)

    print(
        colors.wrap_text(
            f"""
            Invalid input: {response}.
            Regex used to check input: {single_input_re}, {two_input_re}
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
    print("\n******************************************************")
    print(
        colors.wrap_text(
            "Games won:  "
            + f"    {game_stats["wins"]} ({round(
                (game_stats["wins"]/game_stats["games"]) * 100, 4)}%)",
            colors.GREEN
        ))
    print(
        colors.wrap_text(
            "Games drawn:"
            + f"    {game_stats["draws"]} ({round(
                (game_stats["draws"]/game_stats["games"]) * 100, 4)}%)",
            colors.CYAN
        ))
    print(
        colors.wrap_text(
            "Games lost: "
            + f"    {game_stats["losses"]} ({round(
                (game_stats["losses"]/game_stats["games"]) * 100, 4)}%)",
            colors.RED
        ))
    print("******************************************************\n")


def play(environment: TicTacToeGame):

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
        Example input: 3, 0
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
    model_id = model_identifier.strip().replace('.pt', '')
    known_models = {
        "BASELINE":  "TicTacToev2-Baseline-Untrained",
        "NAIVE_2K":  "TicTacToev2-NAIVE-2K",
        "NAIVE_4K":  "TicTacToev2-NAIVE-4K",
        "NAIVE_6K":  "TicTacToev2-NAIVE-6K",
        "NAIVE_8K":  "TicTacToev2-NAIVE-8K",
        "NAIVE_10K": "TicTacToev2-NAIVE-10K",
        "AGENT_1K":  "TicTacToev2-AGENT-1K",
        "AGENT_2K":  "TicTacToev2-AGENT-2K",
        "AGENT_3K":  "TicTacToev2-AGENT-3K",
        "AGENT_4K":  "TicTacToev2-AGENT-4K",
    }

    model_id_upper = model_id.upper()
    for _, (key, value) in enumerate(known_models.items()):
        if model_id_upper == key.upper() \
                or model_id_upper == value.upper():
            return value

    return model_id


if __name__ == "__main__":
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
    play(environment)
