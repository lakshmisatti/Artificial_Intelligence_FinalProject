import matplotlib.pyplot as plt
import numpy as np
from algorithms import QLearningAgent, ApproximateQLearningAgent
from problems import PongGame, PacmanGame

def show_comparison(game_name, episodes=5000):
    print(f"\nTraining comparison for {game_name} ({episodes} episodes each)...")
    
    game = PongGame(headless=True) if game_name == "pong" else PacmanGame(headless=True)
    
    q_rewards = train_agent(game_name, "qlearn", episodes)
    approx_rewards = train_agent(game_name, "approx", episodes)
    
    plt.figure(figsize=(12, 6))
    x_axis = np.linspace(0, episodes, len(q_rewards))
    
    plt.plot(x_axis, q_rewards, label='Q-learning', color='blue')
    plt.plot(x_axis, approx_rewards, label='Approximate Q-learning', color='red')
    
    plt.title(f'{game_name.capitalize()} Performance Comparison\n({episodes} training episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    
    plt.show()  

def train_agent(game_name, agent_type, episodes):
    """Train a single agent and return rewards"""
    game = PongGame(headless=True) if game_name == "pong" else PacmanGame(headless=True)
    
    if agent_type == "qlearn":
        agent = QLearningAgent(
            state_size=game.get_state_space_size(),
            action_size=game.get_num_actions(),
            game_name=game_name
        )
    else:
        agent = ApproximateQLearningAgent(
            num_actions=game.get_num_actions(),
            feature_count=len(game.get_state_features()),
            game_name=game_name
        )
    
    rewards = []
    interval = max(1, episodes // 100)  
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

        if ep % interval == 0 or ep == episodes - 1:
            rewards.append(total_reward)
    
    return rewards