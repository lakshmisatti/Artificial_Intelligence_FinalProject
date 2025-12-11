import argparse
from algorithms import QLearningAgent, ApproximateQLearningAgent
from problems import PongGame, PacmanGame
import time
import os

def get_game(name, headless=False):
    if name == "pong":
        return PongGame(headless=headless)
    elif name == "pacman":
        return PacmanGame(headless=headless)
    else:
        raise ValueError("Unsupported game")

def get_agent(name, game):
    if name == "qlearn":
        return QLearningAgent(
            state_size=game.get_state_space_size(),
            action_size=game.get_num_actions(),
            game_name=type(game).__name__
        )
    elif name == "approx":
        feature_count = len(game.get_state_features())
        return ApproximateQLearningAgent(
            num_actions=game.get_num_actions(),
            feature_count=feature_count,
            game_name=type(game).__name__
        )
    else:
        raise ValueError("Unsupported agent")

def train(agent, game, episodes):
    for ep in range(episodes):
        game.reset()
        state = game.discretize_state() if isinstance(agent, QLearningAgent) else game.get_state_features()
        total_reward = 0

        while not game.is_terminal():
            action = agent.get_action(state)
            reward = game.step(action)
            next_state = game.discretize_state() if isinstance(agent, QLearningAgent) else game.get_state_features()
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            game.render()

        if ep % 500 == 0:
            print(f"Episode {ep}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    agent.save_model()

def test(agent, game, train_mode, game_name):
    agent.load_model()
    test_rewards = []
    
    for ep in range(5):
        game.reset()
        state = game.discretize_state() if isinstance(agent, QLearningAgent) else game.get_state_features()
        total_reward = 0
        start_time = time.time()
        time_limit = 50

        while not game.is_terminal():
            current_time = time.time()
            if current_time - start_time > time_limit:
                print(f"Test Episode {ep} terminated after {time_limit} seconds")
                break
                
            action = agent.get_action(state, exploit=True)
            reward = game.step(action)
            state = game.discretize_state() if isinstance(agent, QLearningAgent) else game.get_state_features()
            total_reward += reward
            game.render()

        test_rewards.append(total_reward)
        print(f"Test Episode {ep}, Total Reward: {total_reward:.2f}")
    
    if not train_mode:
        from comparisons import show_comparison
        show_comparison(game_name)
    
    return test_rewards

def interactive_test_menu():
    print("Select the game:")
    print("1. Pong")
    print("2. Pacman")
    game_input = input("Enter 1 or 2: ").strip()
    game = "pong" if game_input == "1" else "pacman" if game_input == "2" else None
    if not game:
        print("Invalid selection. Exiting.")
        exit(1)

    print("\nSelect the agent:")
    print("1. Q-Learning")
    print("2. Approximate Q-Learning")
    agent_input = input("Enter 1 or 2: ").strip()
    agent = "qlearn" if agent_input == "1" else "approx" if agent_input == "2" else None
    if not agent:
        print("Invalid selection. Exiting.")
        exit(1)

    return game, agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--game", choices=["pong", "pacman"], help="Game to use")
    parser.add_argument("--agent", choices=["qlearn", "approx"], help="Agent to use")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # TRAINING MODE
    if args.train:
        if not args.game or not args.agent:
            print("Error: --game and --agent are required when using --train")
            exit(1)

        game = get_game(args.game, headless=args.headless)
        agent = get_agent(args.agent, game)
        train(agent, game, args.episodes)

    # TESTING MODE (Interactive)
    else:
        game_name, agent_name = interactive_test_menu()
        game = get_game(game_name, headless=args.headless)
        agent = get_agent(agent_name, game)
        test_rewards = test(agent, game, train_mode=False, game_name=game_name)
        print("\nTest completed with rewards:", test_rewards)

if __name__ == "__main__":
    main()
