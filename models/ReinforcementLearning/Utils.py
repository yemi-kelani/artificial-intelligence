import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime

from models.ReinforcementLearning.DeepQ_TicTacToe_v2.DeepQAgent import DeepQAgent
from models.ReinforcementLearning.DeepQ_TicTacToe_v2.TicTacToeGame import TicTacToeGame


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def int2tag(num: int):
    if num <= 0:
        return ""
    elif num < 100:
        return str(num)
    elif num < 1000:
        return str(round(num, -2))
    else:
        return str(int(round(num, -3) / 1000)) + "K"


def train_agent(
    agent: DeepQAgent,
    environment: TicTacToeGame,
    num_episodes: int,
    optimizer,
    criterion,
    device: torch.device,
    save_path: str = "./",
    model_name: str = "",
    save_every: int = -1,
    epsilon_min_value: int = 0.0,
    epsilon_max_value: int = 1.0
):
    if save_every <= 0 or save_every > num_episodes:
        save_every = num_episodes

    epsilon_min_value = max(0.0, epsilon_min_value)
    epsilon_max_value = min(1.0, epsilon_max_value)
    if epsilon_min_value >= epsilon_max_value:
        raise ValueError(
          f"""
          (train_agent:Utils.py) 
          epsilon_min_value cannot exceed epsilon_max_value.
          """)

    if model_name == "":
        time = datetime.now().strftime("%H:%M:%S")
        agent_name = f"{agent.__name__}-{agent.__version__}"
        environment_name = f"{environment.__name__}-{environment.__version__}"
        model_name = f"{agent_name}_{environment_name}_{time}"

    agent.clear_loss_history()
    agent.train()
    reward_history = []
    for episode in range(num_episodes):
        state = environment.get_state()

        steps = 0
        sync_steps = 0
        reward_total = 0
        done = False

        while not done:
            _, mask, indicies = environment.get_valid_moves()
            action = agent.select_action(state, mask, indicies)
            next_state, reward, done = environment.take_action(action)
            
            # EXPERIMENTAL >
            if not done:
                reward, done = environment.resolve_enviornment()
                
            # EXPERIMENTAL <
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state.to(device) if next_state is not None else None
            agent.replay(optimizer, criterion, episode)
            
            steps += 1
            reward_total += reward
            
            # sync the weights of the policy and target networks
            sync_steps += 1
            if agent.use_target_network and sync_steps >= agent.network_sync_rate:
                agent.copy_weights(agent.policy_network, agent.target_network)
                sync_steps = 0
        
        agent.replay(optimizer, criterion, episode)
        
        loss_avg = "n/a"
        if len(agent.get_loss_history()) > 0:
            loss_avg = np.sum(agent.get_loss_history(items_from_back=steps)) / steps

        reward_history.append(reward_total)
        time = datetime.now().strftime("%H:%M:%S")
        print(
            "episode: {}/{}, steps: {}, reward_total: {}, loss_avg: {:.6}, e: {:.4}, time: {}"
            .format(episode + 1, num_episodes, steps, reward_total, loss_avg, agent.epsilon, time)
        )
        
        environment.print_state()
        environment.reset()

        if agent.epsilon < epsilon_min_value:
            agent.epsilon = epsilon_max_value

        if ((episode + 1) % save_every) == 0:
            episode_tag = int2tag(episode + 1)
            checkpoint_name = f"{model_name}-{episode_tag}"
            agent.save_model(save_path, checkpoint_name)

    return reward_history


def test_agent(
    agent,
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
                q_values = agent.forward(state.reshape((1, agent.action_space)))
                q_masked = torch.where(mask != 0, q_values, -1e9)
                action = torch.argmax(q_masked)
                _, reward, done = environment.take_action(action)
                
                # EXPERIMENTAL >
                if not done:
                    reward, done = environment.resolve_enviornment()
                # EXPERIMENTAL <

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

        print("\n")
        print(f"Win rate:  {round((games_won/num_episodes) * 100, 4)}%")
        print(f"Draw rate: {round((games_drawn/num_episodes) * 100, 4)}%")
        print(f"Loss rate: {round((games_lost/num_episodes) * 100, 4)}%")
