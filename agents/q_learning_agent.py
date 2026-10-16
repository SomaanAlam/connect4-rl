import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1):
        """
        Q-Learning Agent for Connect4
        
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
        
        # Initialize Q-table as a dictionary for sparse representation
        self.q_values = {}
        
        # Track learning progress
        self.rewards_history = []
        self.episode_rewards = 0
        self.name = "Q-Learning Agent"
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair"""
        if (state, action) not in self.q_values:
            self.q_values[(state, action)] = 0.0
        return self.q_values[(state, action)]
    
    def get_max_q_value(self, state, valid_actions):
        """Get maximum Q-value for a state"""
        if not valid_actions:
            return 0.0
        return max([self.get_q_value(state, action) for action in valid_actions])
    
    def get_best_action(self, state, valid_actions):
        """Get action with highest Q-value"""
        if not valid_actions:
            return None
        
        best_action = valid_actions[0]
        best_value = self.get_q_value(state, best_action)
        
        for action in valid_actions[1:]:
            value = self.get_q_value(state, action)
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
        """Update Q-values using Q-Learning (off-policy TD learning)"""
        # Convert boards to state representations
        state = self.env.board_to_state(obs.board)
        new_state = self.env.board_to_state(new_obs.board)
        
        # Get valid actions for next state
        valid_actions = self.env.get_valid_actions(new_obs)
        
        # Get current Q-value
        q_value = self.get_q_value(state, action)
        
        # Get max Q-value for next state (off-policy)
        next_max_q = self.get_max_q_value(new_state, valid_actions)
        
        # Q-Learning update
        target = reward + self.gamma * next_max_q
        self.q_values[(state, action)] = q_value + self.alpha * (target - q_value)
        
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
            "q_values_size": len(self.q_values),
            "rewards_history": self.rewards_history
        }