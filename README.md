# Connect 4 Reinforcement Learning

Course project for **Dynamic Programming and Reinforcement Learning**. It compares four RL algorithms learning to play a small Connect 4 variant against random and search-based opponents.

## The task

- **Environment:** Kaggle's `ConnectX` via `kaggle_environments`, on a **4 × 5 board where 3 in a row wins**.
- **Agent role:** the agent always moves first.
- **Reward:** +1 for a win, −1 for a loss, 0 otherwise. Rewards arrive only at the end of a game.
- **Opponents:** `random`, and Kaggle's built-in `negamax` (depth-4 search with random tie-breaking).

## Agents

| Agent | File | Method |
|---|---|---|
| Q-Learning | `agents/q_learning_agent.py` | Off-policy tabular TD, sparse dictionary Q-table keyed on the board string |
| Double Q-Learning | `agents/double_q_learning_agent.py` | Two Q-tables; one selects the best next action, the other evaluates it |
| Sarsa(λ) | `agents/sarsa_lambda_agent.py` | On-policy tabular TD with eligibility traces (λ = 0.9) |
| DQN | `agents/dqn_agent.py` | MLP 60→128→128→5 on a one-hot board, 50k replay buffer, target network, Double-DQN-style targets, gradient clipping, invalid-move masking |

Tabular agents: α = 0.1, γ = 0.95, ε starts at 0.1 and is decayed during training (floor 0.01).
DQN: lr = 1e-3, γ = 0.99, ε 1.0 → 0.05 (×0.995 per episode), batch size 64, target sync every 10 updates.

## Training procedure

A two-phase curriculum (`train_agents.py`):

1. **Phase 1:** play against the random opponent (4,000 episodes for tabular agents, 10,000 for DQN).
2. **Phase 2:** play against negamax (1,000 episodes for tabular agents, 2,500 for DQN).

Every 500 episodes the agent is evaluated over 50 games against each opponent. The final comparison uses 100 games per opponent.

## Results

Final win rates (100 games each, approximate, read from `results/agents_comparison.png`):

| Agent | vs Random | vs Negamax |
|---|---|---|
| Sarsa(λ) | ~0.90 | ~0.96 |
| Q-Learning | ~0.97 | ~0.89 |
| **Double Q-Learning** | ~0.97 | **~0.99** |
| DQN | ~0.81 | ~0.37 |

Observations:

- All tabular agents win 80–100% against random early on, but only 0–20% against negamax until negamax training begins. They then reach 90%+ within about 500 episodes.
- Learned tables are small: roughly 5.9k (Sarsa(λ)), 9.7k (Q-Learning) and 8.6k per table (Double Q) entries, out of a nominal 3²⁰ ≈ 3.5 billion states.
- The DQN performed worst. Its win rate against negamax oscillates, and its training reward collapses from about +0.5 to about −1 when negamax training starts. The cause was not investigated.

Plots are in `results/` (per-agent training curves, reward curves, DQN loss, and the comparison chart). `Figure_1.png` and `Q_learning_random.png` in the root are from earlier 5,000-episode runs.

## Limitations

- **In-distribution evaluation.** Agents are evaluated against the same opponents they trained on, so results may reflect memorising those opponents' habits rather than general play.
- **First-player advantage.** With 3-in-a-row on a 4 × 5 board the first player likely has a large inherent advantage. Agents were only ever trained and tested as player 1.
- **Not purely greedy.** `act()` keeps ε-exploration at evaluation time, so trained agents still play about 1% random moves (about 5% for DQN).
- **Unexplained DQN behaviour.** The replay buffer is dominated by random-opponent games and ε reaches its floor within about 600 episodes. Its training reward against negamax also looks worse than its evaluation win rate.
- **Single run.** Each agent was trained once, with no seeds or confidence intervals.
- **Minor:** `main.py` accepts `--episodes` but ignores it. Saved `.pkl` agents embed the Kaggle environment, so loading them requires `kaggle-environments`, and they should only be loaded from a source you trust (pickle).

## Project structure

```
├── main.py                 # CLI entry point (--train / --use)
├── environment.py          # Connect4Env wrapper around Kaggle ConnectX
├── train_agents.py         # Two-phase training, evaluation, comparison plots
├── use_trained_agents.py   # Loads saved agents and plays 14 matchups
├── agents/                 # Q-Learning, Double Q-Learning, Sarsa(λ), DQN
├── utils/helpers.py        # Save/load and plotting helpers
├── models/                 # Trained agents (.pkl) and DQN weights (.pt)
├── results/                # Training curves and comparison plots
├── requirements.txt
└── .gitignore
```

## Setup and usage

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run everything from the project root:

```bash
python main.py --train   # trains all four agents; overwrites models/ and results/
python main.py --use     # loads saved agents and plays them against each other and the opponents
```

`models/dqn.pkl` (about 35 MB) also contains the DQN replay buffer. `models/dqn_weights.pt` holds just the network weights.
