from kaggle_environments import make
import numpy as np

class Connect4Env:
    def __init__(self, rows=4, columns=5, inarow=3, opponent="random", debug=True):
        self.rows = rows
        self.columns = columns
        self.inarow = inarow
        self.opponent = opponent
        self.debug = debug
        
        # Create Kaggle environment
        self.env = make("connectx", 
                       {"rows": rows, "columns": columns, "inarow": inarow}, 
                       steps=[], 
                       debug=debug)
        
        # State space size (for tabular methods)
        self.state_size = 3**(rows * columns)  # 0, 1, 2 for each cell (empty, player1, player2)
        
        # Action space
        self.action_space = columns
        
    def get_trainer(self, agent_idx=0):
        """Get a trainer instance with the specified opponent"""
        if agent_idx == 0:
            return self.env.train([None, self.opponent])
        else:
            return self.env.train([self.opponent, None])
        
    def reset_trainer(self, trainer):
        """Reset the trainer to start a new episode"""
        return trainer.reset()
    
    def step(self, trainer, action):
        """Take a step in the environment"""
        new_obs, reward, done, info = trainer.step(action)
        return new_obs, reward, done, info
    
    def get_valid_actions(self, obs):
        """Get a list of valid actions (non-full columns)"""
        board = np.array(obs.board).reshape(self.rows, self.columns)
        return [col for col in range(self.columns) if board[0][col] == 0]
    
    def board_to_state(self, board):
        """Convert board to a state representation (for tabular methods)"""
        # This is a simple string representation of the board
        return ''.join(map(str, board))
    
    def render(self, mode="ipython"):
        """Render the environment"""
        try:
            return self.env.render(mode=mode)
        except ModuleNotFoundError:
            # Fallback to ansi mode if ipython is not available
            print("IPython not available, falling back to text mode")
            try:
                return self.env.render(mode="ansi")
            except:
                print("Unable to render the environment. Try installing IPython with 'pip install ipython'")
                return None
    
    def run_agents(self, agent1, agent2):
        """Run a game between two agents"""
        return self.env.run([agent1, agent2])