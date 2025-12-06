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

    for episode in range(num_episodes):
        # Reset environment and ensure proper initial state
        environment.reset(flip_roles=True)
        state = environment.get_state().to(device)
        done, agent_reward_total, steps = False, 0, 0

        while not done:
            # Ensure agent is in training mode
            agent.train()

            _, mask, indices = environment.get_valid_moves()
            action = agent.select_action(state, mask, indices)

            # Agent takes action
            next_state, agent_reward, done = environment.take_action(action)
            next_state = next_state.to(device) if next_state is not None else None

            # If game not over after agent's move, opponent moves
            if not done:
                # Environment/opponent makes move
                opponent_next_state, opponent_result_reward, done = environment.move()
                opponent_next_state = opponent_next_state.to(device) if opponent_next_state is not None else None

                # CRITICAL: If opponent wins after agent's move, agent gets the loss reward
                # This is the key learning signal - the agent's move led to this outcome
                if done:
                    # Game ended after opponent's move - use opponent's result reward
                    # (will be LOSS_REWARD if opponent won, TIE_REWARD if tie)
                    agent_reward = opponent_result_reward
                    next_state = opponent_next_state
                else:
                    next_state = opponent_next_state

            # Store agent's experience with FINAL reward and done state
            agent.remember(state, action, agent_reward, next_state, done)

            # Train agent AFTER experience is stored
            agent.replay(optimizer, criterion, episode)

            state = next_state
            agent_reward_total += agent_reward
            steps += 1
        
        # Update target network AFTER training step
        if agent.use_target_network and (episode + 1) % agent.network_sync_rate == 0:
            agent.copy_weights(agent.policy_network, agent.target_network)

        # log metrics
        reward_history.append(agent_reward_total)
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

        # Calculate loss average more safely
        recent_losses = agent.get_loss_history(items_from_back=max(1, steps))
        loss_avg = np.mean(recent_losses) if recent_losses else 0.0
        
        if (episode + 1) % 10 == 0:  # Print less frequently to reduce clutter
            print(
                f"[{episode+1:04d}/{num_episodes}] Steps: {steps}",
                f"Reward: {agent_reward_total:.2f}, Loss: {loss_avg:.4f}",
                f"Epsilon: {agent.epsilon:.4f}, Role: {environment.positions[environment.agent_role]}",
                f"Win: {100 * wins/(episode + 1):.2f}%",
                f"Draw: {100 * draws/(episode + 1):.2f}%",
                f"Loss: {100 * losses/(episode + 1):.2f}%"
            )

        # Decay epsilon after episode completion
        agent.decay_epsilon(episode)
        agent.epsilon = max(agent.epsilon, agent.epsilon_min)

        if (episode + 1) % 100 == 0:
            agent.save_checkpoint(destination_path=save_path)

        if save_every > 0 and (episode + 1) % save_every == 0:
            tag = int2tag(episode + 1)
            agent.save_model(save_path, f"{model_name}-{tag}")

    return {
        "rewards": reward_history,
        "wins": wins,
        "draws": draws,
        "losses": losses
    }


def test_agent(agent: DeepQAgent, environment: TicTacToeGame, num_episodes: int, print_state: bool = False):
    """Test agent performance with proper evaluation metrics."""
    agent.eval()  # Set to evaluation mode
    results = {
        "X": {"W": 0, "D": 0, "L": 0, "R": 0}, 
        "O": {"W": 0, "D": 0, "L": 0, "R": 0}
    }

    # Track performance statistics
    game_lengths = []
    
    with torch.no_grad():
        for episode in tqdm(range(num_episodes)):
            # Reset environment for each test game
            environment.reset(flip_roles=True)
            role = environment.positions[environment.agent_role]
            state = environment.get_state().to(agent.device)

            done, agent_reward_total, steps = False, 0, 0
            while not done:
                _, mask, _ = environment.get_valid_moves()
                mask = mask.to(agent.device)
                q_values = agent.forward(state.reshape((1, agent.action_space)))
                # Use proper action masking
                q_masked = torch.where(mask != 0, q_values, -1e9)
                action = torch.argmax(q_masked)
                
                # Agent takes action
                next_state, agent_reward, done = environment.take_action(action)
                
                if not done:
                    # Environment/opponent moves
                    next_state, _, done = environment.move()

                state = next_state.to(agent.device) if next_state is not None else None
                agent_reward_total += agent_reward  # Only track agent rewards
                steps += 1
                
                # Prevent infinite games
                if steps > 50:
                    print("Warning: Game exceeded maximum steps, ending as draw")
                    done = True

            game_lengths.append(steps)
            results[role]["R"] += agent_reward_total
            
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
                raise Exception(f"Encountered sync issue between agent and environment.")

            if print_state:
                environment.print_state()

    def summarize(role):
        count = num_episodes // 2 if num_episodes >= 2 else max(1, results[role]['W'] + results[role]['D'] + results[role]['L'])
        if count == 0:
            return
            
        print()
        print(f"Results as {role} ({count} games):")
        print(f"Win rate:  {100 * results[role]['W'] / count:.2f}% ({results[role]['W']}/{count})")
        print(f"Draw rate: {100 * results[role]['D'] / count:.2f}% ({results[role]['D']}/{count})")
        print(f"Loss rate: {100 * results[role]['L'] / count:.2f}% ({results[role]['L']}/{count})")
        print(f"Avg reward: {results[role]['R'] / count:.2f}")
        print()

    # Only summarize roles that were actually played
    total_x_games = results['X']['W'] + results['X']['D'] + results['X']['L']
    total_o_games = results['O']['W'] + results['O']['D'] + results['O']['L']
    
    if total_x_games > 0:
        summarize("X")
    if total_o_games > 0:
        summarize("O")
    
    total_wins = results['X']['W'] + results['O']['W']
    total_games = total_x_games + total_o_games
    
    if total_games > 0:
        print(f"\nOverall Statistics:")
        print(f"Total games: {total_games}")
        print(f"Overall win rate: {100 * total_wins / total_games:.2f}%")
        print(f"Average game length: {np.mean(game_lengths):.1f} moves")
        print(f"Game length std: {np.std(game_lengths):.1f}")
    
    return results
