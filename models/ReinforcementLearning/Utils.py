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
):
    assert agent.epsilon_min <= agent.epsilon_max, "Invalid epsilon range"
    save_every = min(max(0, save_every), num_episodes)

    if not model_name:
        timestamp = datetime.now().strftime("%H%M%S")
        model_name = f"{agent.__class__.__name__}_{environment.__class__.__name__}_{timestamp}"

    agent.clear_loss_history()
    agent.train()

    wins = 0
    draws = 0
    losses = 0
    reward_history = []

    # environment.reset(flip_roles=False)
    for episode in range(num_episodes):
        state = environment.get_state().to(device)
        done, reward_total, steps = False, 0, 0

        while not done:
            _, mask, indices = environment.get_valid_moves()
            action = agent.select_action(state, mask, indices)
            next_state, reward, done = environment.take_action(action)
            next_state = next_state.to(device) if next_state is not None else None
            agent.remember(state, action, reward, next_state, done)
            agent.replay(optimizer, criterion, episode)

            # if game not over, opponent moves, 
            # but we do NOT "remember" that
            if not done:
                next_state, reward, done = environment.move()
                next_state = next_state.to(device) if next_state is not None else None

            state = next_state
            reward_total += reward
            steps += 1
        
        if agent.use_target_network and (episode + 1) % agent.network_sync_rate == 0:
            agent.copy_weights(agent.policy_network, agent.target_network)

        # log metrics
        reward_history.append(reward_total)
        _, _, winner = environment.is_game_over()
        if winner == environment.agent_role:
            wins += 1
        elif winner == environment.role:
            losses += 1
        elif winner is None:
            draws += 1
        else:
            print(f"Invalid winner in endgame state: {winner}")
            environment.print_state()
            raise Exception(f"Encounter sync issue between agent and environment.")

        loss_avg = np.mean(agent.get_loss_history(items_from_back=steps))
        print(
            f"[{episode+1:04d}/{num_episodes}] Steps: {steps}",
            f"Reward: {reward_total:.2f}, Loss: {loss_avg:.4f}",
            f"Epsilon: {agent.epsilon:.4f}, Role: {environment.positions[environment.agent_role]}",
            f"Win: {100 * wins/(episode + 1):.2f}%",
            f"Draw: {100 * draws/(episode + 1):.2f}%",
            f"Loss: {100 * losses/(episode + 1):.2f}%"
        )

        environment.print_state()
        environment.reset(flip_roles=True)
        agent.decay_epsilon(episode)
        agent.epsilon = max(agent.epsilon, agent.epsilon_min)
        print(f"Episode {episode}, Epsilon: {agent.epsilon:.4f}")

        if (episode + 1) % 100 == 0:
            agent.save_checkpoint(destination_path=save_path)

        if (episode + 1) % save_every == 0:
            tag = int2tag(episode + 1)
            agent.save_model(save_path, f"{model_name}-{tag}")
            if hasattr(agent, "memory") and len(agent.memory) > 10000:
                agent.memory = agent.memory[-10000:]

    return {
        "rewards": reward_history,
        "wins": wins,
        "draws": draws,
        "losses": losses
    }


def test_agent(agent: DeepQAgent, environment: TicTacToeGame, num_episodes: int, print_state: bool = False):
    agent.eval()
    results = {
        "X": {"W": 0, "D": 0, "L": 0, "R": 0}, 
        "O": {"W": 0, "D": 0, "L": 0, "R": 0}
    }

    environment.reset(flip_roles=False)
    with torch.no_grad():
        for _ in tqdm(range(num_episodes)):
            role = environment.positions[environment.agent_role]
            state = environment.get_state().to(agent.device)

            done, reward_total = False, 0
            while not done:
                _, mask, _ = environment.get_valid_moves()
                q_values = agent.forward(state.reshape((1, agent.action_space)))
                action = torch.argmax(torch.where(mask != 0, q_values, -1e9))
                next_state, reward, done = environment.take_action(action)
                
                if not done:
                    next_state, reward, done = environment.move()
                    
                state = next_state.to(agent.device) if next_state is not None else None
                reward_total += reward

            results[role]["R"] += reward_total
            _, _, winner = environment.is_game_over()
            if winner == environment.agent_role:
                results[role]["W"] += 1
            elif winner == environment.role:
                results[role]["L"] += 1
            elif winner is None:
                results[role]["D"] += 1
            else:
                print(f"Invalid winner in endgame state: {winner}")
                environment.print_state()
                raise Exception(f"Encounter sync issue between agent and environment.")

            if print_state:
                environment.print_state()
                
            environment.reset(flip_roles=True)

    def summarize(role):
        count = num_episodes // 2
        print()
        print(f"Results as {role}:")
        print(f"Win rate:  {100 * results[role]['W'] / count:.2f}%")
        print(f"Draw rate: {100 * results[role]['D'] / count:.2f}%")
        print(f"Loss rate: {100 * results[role]['L'] / count:.2f}%")
        print(f"Avg reward: {results[role]['R'] / count:.2f}")
        print()

    summarize("X")
    summarize("O")
    print(f"\nOverall win rate: {100 * (results['X']['W'] + results['O']['W']) / num_episodes:.2f}%\n")
