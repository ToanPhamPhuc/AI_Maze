# Maze AI - Deep Q-Learning Implementation

This project implements a Deep Q-Learning (DQL) agent to solve maze navigation problems using your existing maze game.

## Project Structure

```
AI/
├── environment.py      # Maze environment wrapper for RL
├── dqn_model.py        # Deep Q-Network model and agent
├── train.py           # Training script
├── requirements.txt   # Python dependencies
├── models/           # Saved trained models
└── README.md         # This file
```

## How Deep Q-Learning Works

### Core Components:

1. **Environment (`environment.py`)**:
   - Wraps your maze game for RL compatibility
   - Provides `reset()`, `step()`, and `render()` methods
   - Converts maze state to numerical representation

2. **DQN Model (`dqn_model.py`)**:
   - Neural network with 3 fully connected layers
   - Experience replay buffer for stable learning
   - Target network for stable Q-value estimation
   - Epsilon-greedy exploration strategy

3. **Training (`train.py`)**:
   - Main training loop
   - Progress tracking and visualization
   - Model saving and testing

### Learning Process:

1. **State Representation**: 
   - Flattened maze layout (0=path, 1=wall)
   - Player position (normalized coordinates)

2. **Actions**: 
   - 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT

3. **Rewards**:
   - +100: Reaching goal
   - -10: Hitting wall
   - -1: Each step (encourages efficiency)

4. **Training**:
   - Agent explores maze randomly initially (epsilon=1.0)
   - Gradually reduces exploration (epsilon decay)
   - Learns optimal path through experience replay

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your maze game is in the `../GAME/` directory

## Usage

### Training the Agent

```bash
cd AI
python train.py
```

This will:
- Train for 500 episodes
- Save models every 100 episodes
- Render visualization every 50 episodes
- Generate training plots

### Testing a Trained Agent

```python
from train import test_agent
test_agent("models/maze_dqn_final.pth", episodes=10)
```

### Custom Training

```python
from train import train_agent
agent, scores, losses = train_agent(
    episodes=1000,      # Number of episodes
    render_every=100,   # Render frequency
    save_every=200      # Save frequency
)
```

## Hyperparameters

Key parameters in `dqn_model.py`:

- `learning_rate`: 0.001
- `gamma`: 0.99 (discount factor)
- `epsilon`: 1.0 → 0.01 (exploration rate)
- `epsilon_decay`: 0.995
- `memory_size`: 10000 (replay buffer)
- `batch_size`: 64
- `update_target_every`: 1000

## Expected Results

After training, the agent should:
- Find optimal paths to the goal
- Avoid walls consistently
- Complete mazes in minimal steps
- Achieve success rates >90%

## Integration with Game

The trained model can be integrated into your main game to:
- Provide AI opponents
- Show optimal solutions
- Create hint systems
- Demonstrate AI capabilities

## Troubleshooting

1. **CUDA not available**: The code automatically falls back to CPU
2. **Memory issues**: Reduce `memory_size` or `batch_size`
3. **Slow training**: Reduce maze size or use GPU
4. **Poor performance**: Increase training episodes or adjust hyperparameters

## Next Steps

Potential improvements:
- Convolutional layers for better spatial understanding
- Prioritized experience replay
- Dueling DQN architecture
- Multi-agent training
- Transfer learning to larger mazes 