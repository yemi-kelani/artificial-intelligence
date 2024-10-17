import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from DeepQ_TicTacToe_v2 import DeepQAgent

class colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDCOLOR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def wrap_text(text: str, color:str):
        return color + text + colors.ENDCOLOR


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_agent(
    agent: DeepQAgent,
    environment,
    num_episodes: int,
    optimizer,
    criterion,
    device: torch.device,
    save_path: str = "./",
    model_name: str = ""
):
    if model_name == "":
        time = datetime.now().strftime("%H:%M:%S")
        agent_name = f"{agent.__name__}-{agent.__version__}"
        environment_name = f"{environment.__name__}-{environment.__version__}"
        model_name = f"{agent_name}_{environment_name}_{time}"

    agent.train()
    reward_history = []
    for episode in range(num_episodes):
        state = environment.get_state()

        steps = 0
        reward_total = 0
        done = False

        while not done:
            _, mask, indicies = environment.get_valid_moves()
            action = agent.select_action(state, mask, indicies)
            next_state, reward, done = environment.take_action(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state.to(device) if next_state is not None else None
            agent.replay(optimizer, criterion)

            steps += 1
            reward_total += reward

        reward_history.append(reward_total)

        time = datetime.now().strftime("%H:%M:%S")
        print(
            "episode: {}/{}, steps: {}, reward_total: {}, e: {:.2}, time: {}"
            .format(episode + 1, num_episodes, steps, reward_total, agent.epsilon, time)
        )
        environment.print_state()
        environment.reset()

    agent.save_model(save_path, model_name)
    return reward_history


def test_agent(
    agent: DeepQAgent,
    environment,
    num_episodes: int,
    print_state: bool = False
):
    agent.eval()
    with torch.no_grad():

        games_won = 0
        games_drawn = 0
        games_lost = 0

        for episode in tqdm(range(num_episodes)):
            state = environment.get_state()

            steps = 0
            done = False

            while not done:
                _, mask, _ = environment.get_valid_moves()
                q_values = agent.forward(state)
                q_masked = torch.where(mask != 0, q_values, -1000)
                action = torch.argmax(q_masked)
                _, reward, done = environment.take_action(action)

                steps += 1
                if done:

                    if reward == environment.WIN_REWARD:
                        games_won += 1
                    elif reward == environment.TIE_REWARD:
                        games_drawn += 1
                    elif reward == environment.LOSS_REWARD:
                        games_lost += 1

                    break

            if print_state:
                environment.print_state()
            environment.reset()

        print()
        print(f"Win rate:  {round((games_won/num_episodes) * 100, 4)}%")
        print(f"Draw rate: {round((games_drawn/num_episodes) * 100, 4)}%")
        print(f"Loss rate: {round((games_lost/num_episodes) * 100, 4)}%")
