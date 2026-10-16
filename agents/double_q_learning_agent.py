import numpy as np
import random

class DoubleQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1):
        """
        Double Q-Learning Agent for Connect4
        
        Args:
            env: Connect4Env instance
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize two Q-tables for Double Q-Learning
        self.q_values_a = {}
        self.q_values_b = {}
        
        # Track learning progress
        self.rewards_history = []
        self.episode_rewards = 0
        self.name = "Double Q-Learning Agent"
    
    def get_q_value_a(self, state, action):
        """Get Q-value from first table for a state-action pair"""
        if (state, action) not in self.q_values_a:
            self.q_values_a[(state, action)] = 0.0
        return self.q_values_a[(state, action)]
    
    def get_q_value_b(self, state, action):
        """Get Q-value from second table for a state-action pair"""
        if (state, action) not in self.q_values_b:
            self.q_values_b[(state, action)] = 0.0
        return self.q_values_b[(state, action)]
    
    def get_combined_q_value(self, state, action):
        """Get average Q-value from both tables"""
        return (self.get_q_value_a(state, action) + self.get_q_value_b(state, action)) / 2
    
    def get_best_action(self, state, valid_actions):
        """Get action with highest combined Q-value"""
        if not valid_actions:
            return None
        
        best_action = valid_actions[0]
        best_value = self.get_combined_q_value(state, best_action)
        
        for action in valid_actions[1:]:
            value = self.get_combined_q_value(state, action)
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action
    
    def act(self, obs):
        """Choose an action using epsilon-greedy policy"""
        # Convert board to state representation
        state = self.env.board_to_state(obs.board)
        
        # Get valid actions (non-full columns)
        valid_actions = self.env.get_valid_actions(obs)
        
        if not valid_actions:
            return None  # No valid actions
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action (exploration)
            return random.choice(valid_actions)
        else:
            # Greedy action (exploitation)
            return self.get_best_action(state, valid_actions)
    
    def learn(self, obs, new_obs, action, new_action, reward):
        """Update Q-values using Double Q-Learning"""
        # Convert boards to state representations
        state = self.env.board_to_state(obs.board)
        new_state = self.env.board_to_state(new_obs.board)
        
        # Get valid actions for next state
        valid_actions = self.env.get_valid_actions(new_obs)
        
        # Randomly update either Q-table A or Q-table B
        if random.random() < 0.5:
            # Update Q-table A
            best_action = max(valid_actions, key=lambda a: self.get_q_value_a(new_state, a)) if valid_actions else None
            next_q = self.get_q_value_b(new_state, best_action) if best_action is not None else 0
            
            q_value = self.get_q_value_a(state, action)
            target = reward + self.gamma * next_q
            self.q_values_a[(state, action)] = q_value + self.alpha * (target - q_value)
        else:
            # Update Q-table B
            best_action = max(valid_actions, key=lambda a: self.get_q_value_b(new_state, a)) if valid_actions else None
            next_q = self.get_q_value_a(new_state, best_action) if best_action is not None else 0
            
            q_value = self.get_q_value_b(state, action)
            target = reward + self.gamma * next_q
            self.q_values_b[(state, action)] = q_value + self.alpha * (target - q_value)
        
        # Accumulate reward for tracking
        self.episode_rewards += reward
    
    def end_episode(self):
        """Called at the end of an episode to track rewards"""
        self.rewards_history.append(self.episode_rewards)
        self.episode_rewards = 0
    
    def decay_epsilon(self, decay_rate=0.99):
        """Decay exploration rate"""
        self.epsilon = max(0.01, self.epsilon * decay_rate)
    
    def get_stats(self):
        """Get agent statistics"""
        return {
            "name": self.name,
            "q_values_a_size": len(self.q_values_a),
            "q_values_b_size": len(self.q_values_b),
            "rewards_history": self.rewards_history
        }