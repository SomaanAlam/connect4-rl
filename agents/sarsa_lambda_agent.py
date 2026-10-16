import numpy as np
import random

class SarsaLambdaAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1, lambda_=0.9):
        """
        Sarsa(λ) Agent for Connect4
        
        Args:
            env: Connect4Env instance
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            lambda_: Eligibility trace decay rate
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        
        # Initialize Q-table as a dictionary for sparse representation
        self.q_values = {}
        
        # Initialize eligibility traces
        self.traces = {}
        
        # Track learning progress
        self.rewards_history = []
        self.episode_rewards = 0
        self.name = "Sarsa Lambda Agent"
    
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
        """Update Q-values using Sarsa(λ) with eligibility traces"""
        # Convert boards to state representations
        state = self.env.board_to_state(obs.board)
        new_state = self.env.board_to_state(new_obs.board)
        
        # Get Q-values
        q_value = self.get_q_value(state, action)
        
        # Get next Q-value (sarsa is on-policy, so we use the actual next action)
        next_q_value = self.get_q_value(new_state, new_action) if new_action is not None else 0
        
        # Calculate TD error
        td_error = reward + self.gamma * next_q_value - q_value
        
        # Initialize or update eligibility trace for the current state-action pair
        self.traces[(state, action)] = self.traces.get((state, action), 0) + 1
        
        # Update all state-action pairs proportional to their eligibility traces
        for sa_pair in list(self.traces.keys()):
            # Update Q-value
            self.q_values[sa_pair] = self.q_values.get(sa_pair, 0) + \
                                     self.alpha * td_error * self.traces[sa_pair]
            
            # Decay eligibility trace
            self.traces[sa_pair] *= self.gamma * self.lambda_
            
            # Remove trace if it's too small (to save memory)
            if self.traces[sa_pair] < 0.01:
                del self.traces[sa_pair]
        
        # Accumulate reward for tracking
        self.episode_rewards += reward
    
    def end_episode(self):
        """Called at the end of an episode to reset traces and track rewards"""
        self.rewards_history.append(self.episode_rewards)
        self.episode_rewards = 0
        self.traces = {}  # Reset eligibility traces
    
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