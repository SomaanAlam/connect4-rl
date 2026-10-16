from kaggle_environments import make
import numpy as np
from environment import Connect4Env
from utils.helpers import load_agent, create_trained_policy_function
from agents.dqn_agent import DQNAgent


def print_game_result(result):
    """Print the result of a game in a readable format"""
    if not result or len(result) == 0:
        print("Invalid game result")
        return
    
    # Get the last state (final state)
    final_state = result[-1]
    
    if not isinstance(final_state, list) or len(final_state) < 2:
        print("Invalid final state format")
        return
    
    # Extract board and configuration
    # The board will be in the observation of the final state for each player
    board = final_state[0]['observation']['board']
    config = {
        'rows': 4,  # Default values from our environment setup
        'columns': 5
    }
    
    # Try to get dimensions from the environment config if available
    if 'configuration' in final_state[0]:
        if 'rows' in final_state[0]['configuration']:
            config['rows'] = final_state[0]['configuration']['rows']
        if 'columns' in final_state[0]['configuration']:
            config['columns'] = final_state[0]['configuration']['columns']
    
    # Print the board
    board_array = np.array(board).reshape(config['rows'], config['columns'])
    print("\nFinal board state:")
    
    # Print column numbers
    print("  " + " ".join([str(i) for i in range(config['columns'])]))
    
    # Print the board with player 1 as 'X' and player 2 as 'O'
    for row in board_array:
        row_str = []
        for cell in row:
            if cell == 0:
                row_str.append(".")
            elif cell == 1:
                row_str.append("X")
            elif cell == 2:
                row_str.append("O")
        print("| " + " ".join(row_str) + " |")
    
    # Print the result
    if 'reward' in final_state[0]:
        if final_state[0]['reward'] == 1:
            print("\nPlayer 1 (X) won!")
        elif final_state[1]['reward'] == 1:
            print("\nPlayer 2 (O) won!")
        else:
            print("\nThe game ended in a draw.")
    else:
        print("\nGame completed, but no reward information available.")

def main():
    # Create environment
    env = Connect4Env(rows=4, columns=5, inarow=3)
    
    # Load trained agents
    try:
        sarsa_lambda = load_agent("sarsa_lambda")
        q_learning = load_agent("q_learning")
        double_q = load_agent("double_q")
        dqn = load_agent("dqn")
        
        print("Agents loaded successfully!")
    except FileNotFoundError:
        print("Error: Trained agents not found! Please run train_agents.py first.")
        return
    
    # Create policy functions for trained agents
    sarsa_lambda_policy = create_trained_policy_function(sarsa_lambda)
    q_learning_policy = create_trained_policy_function(q_learning)
    double_q_policy = create_trained_policy_function(double_q)
    dqn_policy = create_trained_policy_function(dqn)
    
    
    # Play games with different opponents
    print("\nRunning Sarsa(λ) vs random...")
    result1 = env.env.run([sarsa_lambda_policy, "random"])
    print_game_result(result1)
  
    print("\nRunning Sarsa(λ) vs negamax...")
    result2 = env.env.run([sarsa_lambda_policy, "negamax"])
    print_game_result(result2)
    
    print("\nRunning Q-Learning vs random...")
    result3 = env.env.run([q_learning_policy, "random"])
    print_game_result(result3)
    
    print("\nRunning Q-Learning vs negamax...")
    result4 = env.env.run([q_learning_policy, "negamax"])
    print_game_result(result4)

    print("\nRunning Double Q-Learning vs random...")
    result5 = env.env.run([double_q_policy, "random"])
    print_game_result(result5)
    
    print("\nRunning Double Q-Learning vs negamax...")
    result6=env.env.run([double_q_policy, "negamax"])
    print_game_result(result6)

    print("\nRunning DQN vs random...")
    result10 = env.env.run([dqn_policy, "random"])
    print_game_result(result10)
    
    print("\nRunning DQN vs negamax...")
    result11=env.env.run([dqn_policy, "negamax"])
    print_game_result(result11)

    print("\nRunning DQN vs Sarsa(λ)...")
    result12=env.env.run([dqn_policy, sarsa_lambda_policy])
    print_game_result(result12)

    print("\nRunning DQN vs Q-Learning...")
    result13=env.env.run([dqn_policy, q_learning_policy])
    print_game_result(result13)

    print("\nRunning DQN vs Double Q-Learning...")
    result14=env.env.run([dqn_policy, double_q_policy])
    print_game_result(result14)


    print("\nRunning Q-Learning vs Sarsa(λ)...")
    result7=env.env.run([q_learning_policy, sarsa_lambda_policy])
    print_game_result(result7)

    print("\nRunning Double Q-Learning vs Sarsa(λ)...")
    result8=env.env.run([double_q_policy, sarsa_lambda_policy])
    print_game_result(result8)

    print("\nRunning Double Q-Learning vs Q-Learning...")
    result9=env.env.run([double_q_policy, q_learning_policy])
    print_game_result(result9)
    

    print("\nDone!")

if __name__ == "__main__":
    main()