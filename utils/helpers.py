import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def save_agent(agent, filename):
    """Save agent's policy/q-values to a file"""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    path = f"models/{filename}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(agent, f)
    print(f"Agent saved to {path}")

def load_agent(filename):
    """Load agent from a file"""
    path = f"models/{filename}.pkl"
    with open(path, 'rb') as f:
        agent = pickle.load(f)
    print(f"Agent loaded from {path}")
    return agent

def plot_rewards(rewards, window=100, filename=None):
    """Plot rewards over episodes with moving average"""
    if not os.path.exists('results'):
        os.makedirs('results')
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, color='blue')
    
    # Compute moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, color='red')
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Rewards over Training Episodes')
    plt.grid(True)
    
    if filename:
        plt.savefig(f"results/{filename}.png")
    
    plt.show()

def plot_win_rates(win_rates, labels, filename=None):
    """Plot win rates comparison between agents"""
    if not os.path.exists('results'):
        os.makedirs('results')
    
    plt.figure(figsize=(10, 6))
    x = range(len(labels))
    plt.bar(x, win_rates, width=0.6)
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.xlabel('Opponent')
    plt.ylabel('Win Rate')
    plt.title('Agent Performance Against Different Opponents')
    plt.grid(True, axis='y')
    
    if filename:
        plt.savefig(f"results/{filename}.png")
    
    plt.show()

def create_trained_policy_function(agent):
    """Create a function that uses the trained agent to make decisions"""
    def trained_policy(obs):
        return agent.act(obs)
    return trained_policy