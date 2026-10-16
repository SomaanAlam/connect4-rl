import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from environment import Connect4Env
from agents.sarsa_lambda_agent import SarsaLambdaAgent
from agents.q_learning_agent import QLearningAgent
from agents.double_q_learning_agent import DoubleQLearningAgent
from agents.dqn_agent import DQNAgent
from utils.helpers import save_agent, load_agent, plot_rewards, plot_win_rates, create_trained_policy_function

def train_agent(agent, env, num_random=3500, num_negamax=1500, eval_interval=500, verbose=True):
    """
    Train agent with mixed strategy: first against random opponent, then against negamax
    
    Args:
        agent: The agent to train
        env: The environment
        num_random: Number of episodes to train against random opponent (Phase 1)
        num_negamax: Number of episodes to train against negamax opponent (Phase 2)
        eval_interval: How often to evaluate the agent
        verbose: Whether to print progress
    """
    # Training stats
    eval_history = []
    start_time = time.time()
    total_episodes = num_random + num_negamax
    
    # PHASE 1: Train against random opponent
    # ======================================
    env.opponent = "random"
    trainer = env.get_trainer()
    
    print(f"Phase 1: Training against random opponent for {num_random} episodes...")
    for episode in tqdm(range(1, num_random + 1)):
        # Reset environment for new episode
        obs = env.reset_trainer(trainer)
        done = False
        
        # Get initial action
        action = agent.act(obs)
        
        # Episode loop - play until game ends
        while not done:
            # Take action and observe result
            new_obs, reward, done, info = env.step(trainer, action)
            
            # Select next action (if not done)
            if done:
                new_action = None
            else:
                new_action = agent.act(new_obs)
                
            # Update agent's knowledge
            agent.learn(obs, new_obs, action, new_action, reward)
            
            # Move to next state
            obs = new_obs
            action = new_action
        
        # End of episode - update agent and collect rewards
        agent.end_episode()  # Note: DQN agent handles its own epsilon decay in this method
        
        # Decay exploration rate periodically
        if episode % 100 == 0 and hasattr(agent, 'decay_epsilon'):
            agent.decay_epsilon(0.95)
        
        # Evaluate agent performance periodically
        if episode % eval_interval == 0:
            # Test against both opponent types
            random_win_rate = evaluate_agent(agent, env, "random", num_games=50)
            negamax_win_rate = evaluate_agent(agent, env, "negamax", num_games=50)
            eval_history.append((random_win_rate, negamax_win_rate))
            
            if verbose:
                print(f"Episode {episode}/{total_episodes} - Win rates: Random: {random_win_rate:.2f}, Negamax: {negamax_win_rate:.2f}")
    
    # PHASE 2: Train against negamax opponent
    # ======================================
    env.opponent = "negamax"
    trainer = env.get_trainer()
    
    print(f"Phase 2: Training against negamax opponent for {num_negamax} episodes...")
    for episode in tqdm(range(num_random + 1, total_episodes + 1)):
        # Reset environment for new episode
        obs = env.reset_trainer(trainer)
        done = False
        
        # Get initial action
        action = agent.act(obs)
        
        # Episode loop - play until game ends
        while not done:
            # Take action and observe result
            new_obs, reward, done, info = env.step(trainer, action)
            
            # Select next action (if not done)
            if done:
                new_action = None
            else:
                new_action = agent.act(new_obs)
                
            # Update agent's knowledge
            agent.learn(obs, new_obs, action, new_action, reward)
            
            # Move to next state
            obs = new_obs
            action = new_action
        
        # End of episode - update agent and collect rewards
        agent.end_episode()  # Note: DQN agent handles its own epsilon decay in this method
        
        # Decay exploration rate periodically (more aggressively in Phase 2)
        if episode % 100 == 0 and hasattr(agent, 'decay_epsilon'):
            agent.decay_epsilon(0.9)  # Slightly more aggressive decay
        
        # Evaluate agent performance periodically
        if episode % eval_interval == 0:
            # Test against both opponent types
            random_win_rate = evaluate_agent(agent, env, "random", num_games=50)
            negamax_win_rate = evaluate_agent(agent, env, "negamax", num_games=50)
            eval_history.append((random_win_rate, negamax_win_rate))
            
            if verbose:
                print(f"Episode {episode}/{total_episodes} - Win rates: Random: {random_win_rate:.2f}, Negamax: {negamax_win_rate:.2f}")
    
    # Training complete
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot evaluation history to show learning progress
    if eval_history:
        # Create checkpoints for x-axis (evaluation episodes)
        checkpoints = []
        for i in range(0, total_episodes + 1, eval_interval):
            if i >= eval_interval:
                checkpoints.append(i)
        
        # Extract win rates against each opponent type
        random_rates = [r for r, _ in eval_history]
        negamax_rates = [n for _, n in eval_history]
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(checkpoints[:len(random_rates)], random_rates, 'b-', label='vs Random')
        plt.plot(checkpoints[:len(negamax_rates)], negamax_rates, 'r-', label='vs Negamax')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate')
        plt.title(f'{agent.name} Performance During Mixed Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results/{agent.name.replace(' ', '_').lower()}_training.png")
    
    return eval_history

def evaluate_agent(agent, env, opponent="random", num_games=100):
    """Evaluate agent performance against an opponent"""
    # Save current opponent
    current_opponent = env.opponent
    
    # Set opponent for evaluation
    env.opponent = opponent
    
    wins = 0
    opponent_wins = 0
    draws = 0
    
    for _ in range(num_games):
        trainer = env.get_trainer()
        obs = env.reset_trainer(trainer)
        done = False
        
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(trainer, action)
            
            if done:
                if reward == 1:  # Agent won
                    wins += 1
                elif reward == -1:  # Opponent won
                    opponent_wins += 1
                else:  # Draw
                    draws += 1
    
    win_rate = wins / num_games
    
    # Restore original opponent
    env.opponent = current_opponent
    
    return win_rate

def compare_agents(env, agents, opponents, num_games=100):
    """Compare multiple agents against multiple opponents"""
    results = {}
    
    for agent_name, agent in agents.items():
        agent_results = []
        
        for opponent in opponents:
            win_rate = evaluate_agent(agent, env, opponent, num_games)
            agent_results.append(win_rate)
            print(f"{agent_name} vs {opponent}: Win rate = {win_rate:.2f}")
        
        results[agent_name] = agent_results
    
    # Plot results
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.2
    index = np.arange(len(opponents))
    
    for i, (agent_name, win_rates) in enumerate(results.items()):
        ax.bar(index + i*bar_width, win_rates, bar_width, label=agent_name)
    
    ax.set_xlabel('Opponent')
    ax.set_ylabel('Win Rate')
    ax.set_title('Agents Performance Comparison')
    ax.set_xticks(index + bar_width * (len(agents) - 1) / 2)
    ax.set_xticklabels(opponents)
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig("results/agents_comparison.png")
    plt.show()
    
    return results

def main():
    # Create environment
    env = Connect4Env(rows=4, columns=5, inarow=3, opponent="random")
    
    # Create agents
    sarsa_lambda = SarsaLambdaAgent(env, alpha=0.1, gamma=0.95, epsilon=0.1, lambda_=0.9)
    q_learning = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon=0.1)
    double_q = DoubleQLearningAgent(env, alpha=0.1, gamma=0.95, epsilon=0.1)
    dqn = DQNAgent(env, hidden_dims=[128, 128], lr=0.001, gamma=0.99, epsilon=1.0, 
                  epsilon_min=0.05, epsilon_decay=0.995, batch_size=64, target_update=10)
    
    # Train agents with mixed strategy (random + negamax)
    print("Training Sarsa(λ) agent...")
    train_agent(sarsa_lambda, env, num_random=4000, num_negamax=1000)
    
    print("Training Q-Learning agent...")
    train_agent(q_learning, env, num_random=4000, num_negamax=1000)
    
    print("Training Double Q-Learning agent...")
    train_agent(double_q, env, num_random=4000, num_negamax=1000)
    
    print("Training DQN agent...")
    train_agent(dqn, env, num_random=10000, num_negamax=2500)
    
    # Plot learning curves
    plot_rewards(sarsa_lambda.rewards_history, window=100, filename="sarsa_lambda_rewards")
    plot_rewards(q_learning.rewards_history, window=100, filename="q_learning_rewards")
    plot_rewards(double_q.rewards_history, window=100, filename="double_q_rewards")
    plot_rewards(dqn.rewards_history, window=100, filename="dqn_rewards")
    
    # Plot DQN loss if available
    if hasattr(dqn, 'losses') and dqn.losses:
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(dqn.losses)[::100])  # Plot every 100th loss value to reduce noise
        plt.xlabel('Training Steps (x100)')
        plt.ylabel('Loss')
        plt.title('DQN Training Loss')
        plt.grid(True)
        plt.savefig("results/dqn_loss.png")
        plt.show()
    
    # Compare agents
    agents = {
        "Sarsa(λ)": sarsa_lambda,
        "Q-Learning": q_learning,
        "Double Q-Learning": double_q,
        "DQN": dqn
    }
    
    opponents = ["random", "negamax"]
    compare_agents(env, agents, opponents, num_games=100)
    
    # Save trained agents
    save_agent(sarsa_lambda, "sarsa_lambda")
    save_agent(q_learning, "q_learning")
    save_agent(double_q, "double_q")
    save_agent(dqn, "dqn")
    
    # Save DQN model weights separately
    dqn.save("models/dqn_weights.pt")
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()