import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# DQN Neural Network Model
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128]):
        super(DQNetwork, self).__init__()
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env, hidden_dims=[128, 128], lr=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.05, epsilon_decay=0.995, batch_size=64, target_update=10):
        """
        DQN Agent for Connect4 using function approximation
        
        Args:
            env: Connect4Env instance
            hidden_dims: List of hidden layer dimensions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            batch_size: Size of mini-batch for training
            target_update: How often to update target network
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # State and action dimensions
        self.state_dim = self.env.rows * self.env.columns * 3  # One-hot encoded board state
        self.action_dim = self.env.columns
        
        # Q-Networks (main and target)
        self.q_network = DQNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_network = DQNetwork(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is in evaluation mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=50000)
        
        # Training counters
        self.train_step = 0
        
        # Training history
        self.rewards_history = []
        self.losses = []
        self.episode_reward = 0
        
        self.name = "DQN Agent"
    
    def state_to_tensor(self, board):
        """Convert Connect4 board to tensor with one-hot encoding"""
        # One-hot encode: 0 -> empty, 1 -> player 1, 2 -> player 2
        state = np.zeros((len(board), 3), dtype=np.float32)
        
        for i, cell in enumerate(board):
            if cell == 0:
                state[i, 0] = 1.0  # Empty
            elif cell == 1:
                state[i, 1] = 1.0  # Player 1
            elif cell == 2:
                state[i, 2] = 1.0  # Player 2
        
        # Flatten and convert to tensor
        state = state.flatten()
        return torch.FloatTensor(state).to(self.device)
    
    def act(self, obs):
        """Choose an action using epsilon-greedy policy"""
        # Get valid actions
        valid_actions = self.env.get_valid_actions(obs)
        
        if not valid_actions:
            return None  # No valid actions
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action (exploration)
            return random.choice(valid_actions)
        else:
            # Greedy action (exploitation)
            state_tensor = self.state_to_tensor(obs.board).unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():  # No gradient calculation needed
                q_values = self.q_network(state_tensor)
            
            # Create a mask for valid actions (-inf for invalid actions)
            mask = torch.ones(self.action_dim) * float('-inf')
            for action in valid_actions:
                mask[action] = 0
            
            # Apply mask and get best action
            masked_q_values = q_values.squeeze(0) + mask.to(self.device)
            action = torch.argmax(masked_q_values).item()
            
            return action
    
    def learn(self, obs, new_obs, action, new_action, reward):
        """Store experience and update networks"""
        # Convert board states to tensors
        state = self.state_to_tensor(obs.board)
        next_state = self.state_to_tensor(new_obs.board)
        done = new_action is None
        
        # Store transition in replay buffer
        self.memory.add(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
        
        # Accumulate episode reward
        self.episode_reward += reward
        
        # Only train if we have enough samples
        if len(self.memory) >= self.batch_size:
            self._update_q_network()
            
            # Update target network periodically
            self.train_step += 1
            if self.train_step % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _update_q_network(self):
        """Update Q-network using a batch of experiences"""
        # Sample mini-batch from replay buffer
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        curr_q_values = self.q_network(states).gather(1, actions).squeeze(1)
        
        # Compute next Q values with target network (Double DQN)
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
            # Get values from target network
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            # Compute targets
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(curr_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Store loss for tracking
        self.losses.append(loss.item())
    
    def end_episode(self):
        """Called at the end of an episode"""
        # Store episode reward
        self.rewards_history.append(self.episode_reward)
        self.episode_reward = 0
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_stats(self):
        """Get agent statistics"""
        return {
            "name": self.name,
            "rewards_history": self.rewards_history,
            "losses": self.losses if hasattr(self, 'losses') else [],
            "epsilon": self.epsilon
        }
    
    def save(self, filepath):
        """Save model weights and parameters"""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights and parameters"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        print(f"Model loaded from {filepath}")