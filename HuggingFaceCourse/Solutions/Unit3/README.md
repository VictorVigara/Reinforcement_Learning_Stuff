# Unit 3: Deep Q-Learning with Atari Games using RL Baselines3 Zoo

This directory contains experiments training Deep Q-Learning (DQN) agents on Atari games using the RL Baselines3 Zoo framework.

## Experiments Overview

### Experiment 1: Training DQN on Space Invaders
- **Environment**: SpaceInvadersNoFrameskip-v4
- **Algorithm**: Deep Q-Learning (DQN)
- **Training Steps**: 1,000,000 timesteps
- **Framework**: RL Baselines3 Zoo with Stable-Baselines3

### Experiment 2: Evaluating Pre-trained Agent
- **Environment**: BeamRiderNoFrameskip-v4
- **Model Source**: Stable-Baselines3 model zoo (sb3/dqn-BeamRiderNoFrameskip-v4)
- **Purpose**: Demonstrate loading and evaluating pre-trained models from Hugging Face Hub

## Deep Q-Learning (DQN) Algorithm

### Overview
Deep Q-Learning combines Q-Learning with deep neural networks to handle high-dimensional state spaces like raw pixel inputs from Atari games. The algorithm learns a Q-function Q(s,a) that estimates the expected cumulative reward for taking action 'a' in state 's'.

### Key Components

1. **Deep Q-Network (DQN)**
   - A convolutional neural network that approximates the Q-function
   - Takes preprocessed game frames as input
   - Outputs Q-values for each possible action

2. **Experience Replay Buffer**
   - Stores past experiences (state, action, reward, next_state, done)
   - Samples random mini-batches for training
   - Breaks correlation between consecutive samples
   - Improves sample efficiency by reusing experiences

3. **Target Network**
   - A separate network with frozen parameters
   - Used to compute target Q-values for stability
   - Periodically updated with the main network's weights
   - Prevents moving target problem during training

4. **Epsilon-Greedy Exploration**
   - Balances exploration vs exploitation
   - Starts with high epsilon (random actions)
   - Gradually decreases to favor learned policy
   - Ensures agent explores the environment sufficiently

### DQN Algorithm Steps

1. Initialize replay buffer D and Q-network with random weights
2. For each episode:
   - Observe initial state s
   - For each timestep:
     - Select action using epsilon-greedy: a = argmax Q(s,a) with probability 1-ε, else random
     - Execute action, observe reward r and next state s'
     - Store transition (s, a, r, s', done) in replay buffer
     - Sample random mini-batch from replay buffer
     - Compute target: y = r + γ * max Q_target(s', a') for non-terminal s'
     - Update Q-network by minimizing loss: (y - Q(s,a))²
     - Every C steps, update target network: Q_target ← Q
     - s ← s'

## Hyperparameters Configuration

The DQN agent was configured with the following hyperparameters (defined in `dqn.yml`):

### Environment Processing
```yaml
env_wrapper: stable_baselines3.common.atari_wrappers.AtariWrapper
frame_stack: 4
```
- **AtariWrapper**: Preprocesses Atari frames (grayscale, resize to 84x84, frame skipping)
- **frame_stack**: Stacks 4 consecutive frames to capture motion information

### Network Architecture
```yaml
policy: 'CnnPolicy'
```
- **CnnPolicy**: Convolutional Neural Network policy for processing image inputs
- Architecture: 3 convolutional layers followed by fully connected layers

### Training Parameters
```yaml
n_timesteps: 1e6              # Total training timesteps
buffer_size: 100000           # Experience replay buffer size
learning_rate: 1e-4           # Adam optimizer learning rate
batch_size: 32                # Mini-batch size for training
learning_starts: 100000       # Steps before training begins
target_update_interval: 1000  # Frequency of target network updates
train_freq: 4                 # Training frequency (every 4 steps)
gradient_steps: 1             # Number of gradient steps per update
optimize_memory_usage: False  # Memory optimization flag
```

### Exploration Parameters
```yaml
exploration_fraction: 0.1     # Fraction of training for epsilon decay
exploration_final_eps: 0.01   # Final epsilon value
```
- **exploration_fraction**: Epsilon decays from 1.0 to 0.01 over first 10% of training (100K steps)
- **exploration_final_eps**: Minimum exploration rate maintained after decay

### Parameter Explanations

**buffer_size (100,000)**: Stores the last 100K transitions. Larger buffers provide more diverse samples but require more memory.

**learning_starts (100,000)**: Agent explores randomly for 100K steps to fill replay buffer with diverse experiences before training begins.

**target_update_interval (1,000)**: Target network updated every 1,000 steps to provide stable learning targets while allowing the main network to improve.

**train_freq (4)**: Performs one gradient update every 4 environment steps, balancing learning efficiency with computational cost.

**batch_size (32)**: Number of transitions sampled from replay buffer for each gradient update. Larger batches provide more stable gradients but increase computation.

## Training Results

### Space Invaders Performance

The agent was trained across multiple runs, with the best performance achieved in run #4:

**Training Progress** (evaluated every 25,000 steps):

| Timesteps | Mean Reward | Episode Length |
|-----------|-------------|----------------|
| 25,000    | 162 ± 52    | ~2,400         |
| 100,000   | 264 ± 45    | ~2,800         |
| 250,000   | 119 ± 54    | ~2,000         |
| 500,000   | 390 ± 167   | ~3,300         |
| 750,000   | 650 ± 346   | ~4,100         |
| 1,000,000 | 693 ± 356   | ~4,400         |

**Best Evaluation Results** (at 1M timesteps):
- Mean reward: 693.0
- Std reward: 356.3
- Max reward: 1,330
- Min reward: 400
- Mean episode length: 4,447 steps

### Learning Curve Analysis

The agent showed clear learning progression:

1. **Early Training (0-100K steps)**: Random exploration phase, rewards around 150-300
2. **Initial Learning (100K-500K steps)**: Inconsistent performance as agent learns basic strategies
3. **Skill Development (500K-750K steps)**: Significant improvement, mean rewards climbing to 400-600
4. **Advanced Play (750K-1M steps)**: Consistent high performance, achieving 600-1,300 points

The high standard deviation in later stages indicates the agent learned different strategies with varying success rates, which is typical for Space Invaders where game difficulty increases with progression.

## Evaluation Protocol

### Space Invaders Agent
```bash
python -m rl_zoo3.enjoy \
  --algo dqn \
  --env SpaceInvadersNoFrameskip-v4 \
  --no-render \
  --n-timesteps 5000 \
  --folder logs/
```

### Pre-trained BeamRider Agent
```bash
# Download model from Hub
python -m rl_zoo3.load_from_hub \
  --algo dqn \
  --env BeamRiderNoFrameskip-v4 \
  -orga sb3 \
  -f rl_trained/

# Evaluate for 5000 timesteps
python -m rl_zoo3.enjoy \
  --algo dqn \
  --env BeamRiderNoFrameskip-v4 \
  -n 5000 \
  -f rl_trained/ \
  --no-render
```

## Files Structure

```
Solutions/Unit3/
├── README.md                    # This file
├── dqn.yml                      # Hyperparameter configuration
└── logs/
    └── dqn/
        ├── SpaceInvadersNoFrameskip-v4_1/  # Training run 1
        ├── SpaceInvadersNoFrameskip-v4_2/  # Training run 2
        ├── SpaceInvadersNoFrameskip-v4_3/  # Training run 3
        └── SpaceInvadersNoFrameskip-v4_4/  # Training run 4 (best)
            ├── best_model.zip              # Best performing checkpoint
            ├── SpaceInvadersNoFrameskip-v4.zip  # Final model
            ├── evaluations.npz             # Evaluation results
            ├── 0.monitor.csv               # Training episode logs
            └── SpaceInvadersNoFrameskip-v4/
                ├── args.yml                # Training arguments
                ├── config.yml              # Environment config
                └── command.txt             # Training command
```

## Key Insights

1. **Importance of Exploration**: The 100K learning_starts period was crucial for filling the replay buffer with diverse experiences before training.

2. **Stability through Target Network**: Updating the target network every 1,000 steps provided stable learning targets while allowing the main Q-network to improve.

3. **Frame Stacking**: Stacking 4 frames was essential for the agent to understand motion and velocity of enemies and projectiles.

4. **Sample Efficiency**: Experience replay allowed the agent to learn from past experiences multiple times, significantly improving sample efficiency.

5. **Training Duration**: 1M timesteps (~90 minutes on GPU) provided good results, though longer training (10M steps) could potentially improve performance further.

## Reproducing Results

To reproduce the training:

```bash
# Train Space Invaders agent
python -m rl_zoo3.train \
  --algo dqn \
  --env SpaceInvadersNoFrameskip-v4 \
  -f logs/ \
  -c dqn.yml

# Evaluate trained agent
python -m rl_zoo3.enjoy \
  --algo dqn \
  --env SpaceInvadersNoFrameskip-v4 \
  --no-render \
  --n-timesteps 5000 \
  --folder logs/
```

## References

- [RL Baselines3 Zoo Documentation](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Stable-Baselines3 DQN Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- [Original DQN Paper](https://arxiv.org/abs/1312.5602) - "Playing Atari with Deep Reinforcement Learning" by Mnih et al.
- [Human-level Control Paper](https://www.nature.com/articles/nature14236) - "Human-level control through deep reinforcement learning" by Mnih et al. (Nature, 2015)
- [Hugging Face Deep RL Course - Unit 3](https://huggingface.co/deep-rl-course/unit3/introduction)

## Future Improvements

Potential enhancements to explore:

1. **Extended Training**: Train for 10M timesteps for better convergence
2. **DQN Extensions**: Implement Double-DQN, Dueling-DQN, or Prioritized Experience Replay
3. **Hyperparameter Tuning**: Use Optuna to optimize learning_rate, buffer_size, and batch_size
4. **Additional Environments**: Test on other Atari games (Breakout, Pong, BeamRider)
5. **Ensemble Methods**: Train multiple agents and combine their policies
