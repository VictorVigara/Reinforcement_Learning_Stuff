# Reinforcement Learning Training - Learning Guide

This guide explains the monitoring tools and experiments you can try to understand RL training better.

## 🎯 What's Been Added to Unit1.py

### 1. **TensorBoard Logging**
```python
tensorboard_log=tensorboard_log_dir
```
**What it does:** Logs training metrics in real-time to visualize learning progress.

**How to use:**
```bash
# In a separate terminal, run:
tensorboard --logdir Solutions/Unit1/tensorboard_logs
# Then open http://localhost:6006 in your browser
```

**What you'll see:**
- `rollout/ep_len_mean`: Average episode length (shorter = agent dies faster)
- `rollout/ep_rew_mean`: Average episode reward (higher = better performance)
- `train/loss`: Policy and value function losses
- `train/entropy_loss`: Exploration measure (decreases over time)
- `train/policy_gradient_loss`: How much the policy is changing
- `train/value_loss`: How well the agent predicts future rewards

### 2. **EvalCallback**
```python
eval_callback = EvalCallback(eval_env, eval_freq=5000, ...)
```
**What it does:** Periodically evaluates the agent on fresh episodes and saves the best model.

**Benefits:**
- Tracks true performance (not just training rewards)
- Automatically saves best performing model to `best_model/`
- Creates evaluation logs in `eval_logs/evaluations.npz`

### 3. **CheckpointCallback**
```python
checkpoint_callback = CheckpointCallback(save_freq=10000, ...)
```
**What it does:** Saves model snapshots during training.

**Use cases:**
- Resume training if interrupted
- Compare models at different training stages
- Recover if final model overfits

## 🔄 Understanding the Training Loop: Steps, Batches, and Episodes

### How PPO Training Works (The Complete Picture)

PPO training happens in a cycle of **COLLECT → LEARN → REPEAT**:

```
┌─────────────────────────────────────────────────────┐
│ ROLLOUT PHASE (Data Collection)                    │
│                                                     │
│ Each of 16 envs runs for n_steps (1024)            │
│ Total transitions collected: 16 × 1024 = 16,384    │
│                                                     │
│ Episodes happen naturally during this:             │
│ Env 1: [Episode 1---300][Ep 2--250][Ep 3----474]   │
│ Env 2: [Episode 1---280][Ep 2--290][Ep 3----454]   │
│ ...                                                 │
│ Approx 48-80 complete episodes in this rollout     │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ LEARNING PHASE (Policy Update)                     │
│                                                     │
│ Buffer: 16,384 transitions                          │
│                                                     │
│ Epoch 1: Shuffle → Split into batches of 64        │
│          → 256 gradient updates (16384/64)          │
│ Epoch 2: Shuffle → Split into batches of 64        │
│          → 256 gradient updates                     │
│ Epoch 3: Shuffle → Split into batches of 64        │
│          → 256 gradient updates                     │
│ Epoch 4: Shuffle → Split into batches of 64        │
│          → 256 gradient updates                     │
│                                                     │
│ Total: 1,024 gradient updates per learning phase   │
└─────────────────────────────────────────────────────┘
                      ↓
        Throw away data, start new rollout
```

### Key Relationships

**1. Episodes vs n_steps: They are INDEPENDENT**

- `n_steps` = How long each env runs before we stop to learn
- Episode length = How long until the environment naturally ends (crash, goal, timeout)

**Episodes can:**
- End before n_steps (agent crashes early)
- Span multiple rollouts (long episodes continue across collection phases)
- Vary in length (some episodes short, some long)

**Example with n_steps=1024:**
```
Steps:  0--------300-------600--------1024--------1400------2048
        |  Ep 1   |  Ep 2  |   Ep 3   |   Ep 4    |  Ep 5  |
        |_____Rollout 1______________|_____Rollout 2_________|

Episode 1: 300 steps (ends, resets)
Episode 2: 300 steps (ends, resets)
Episode 3: 424 steps (ends, resets)
Episode 4: 376 steps (ends, resets)
Episode 5: Continues into next rollout...
```

**2. Total Transitions Per Update**
```
Total transitions = n_steps × num_envs
Your setup: 1024 × 16 = 16,384 transitions
```

**3. Gradient Updates Per Learning Phase**
```
Updates per epoch = total_transitions / batch_size
Total updates = updates_per_epoch × n_epochs

Your setup: (16,384 / 64) × 4 = 1,024 gradient updates
```

### Rule of Thumb for Setting n_steps

**Guideline: n_steps should be 2-5× your typical episode length**

Why?
- Ensures you get multiple complete episodes per rollout
- Provides diverse data (different starting states)
- Good temporal coverage for advantage estimation

**Examples:**

| Environment | Typical Episode Length | Recommended n_steps | Reasoning |
|-------------|----------------------|-------------------|-----------|
| CartPole-v1 | ~200 steps | 512-1024 | 2-5× coverage |
| LunarLander-v2 | 200-300 steps | 1024-2048 | Current setup ✓ |
| Atari Pong | 1000-2000 steps | 2048-4096 | Longer episodes |
| MuJoCo Walker | 1000 steps | 2048 | Standard |
| Custom short task | 50 steps | 128-256 | Don't need much |

**If n_steps is TOO SMALL:**
- Miss long-term dependencies
- Poor advantage estimates
- Inefficient learning

**If n_steps is TOO LARGE:**
- Slower updates (collect more before learning)
- Memory issues
- Stale policy (collecting with old policy too long)

### Simulation Step Frequency - IMPORTANT!

**Yes, this matters a lot for understanding your environment!**

Simulation frequency tells you **real time per step**:

```
Step frequency = steps per second (Hz)
Time per step = 1 / frequency

Example: 50 Hz simulation
→ Each step = 1/50 = 0.02 seconds = 20ms
→ 100 steps = 2 seconds of simulated time
```

**Why this matters:**

1. **Understanding Episode Length**
   ```
   If your task takes ~10 real-world seconds
   And your simulation runs at 50 Hz
   → Episode will be ~500 steps
   → Set n_steps = 1024-2048
   ```

2. **Setting Realistic Expectations**
   ```
   Robot arm task:
   - Real task duration: 5 seconds
   - Simulation: 100 Hz
   - Expected episode length: 500 steps
   - Max episode steps: 1000 (allow some exploration time)
   ```

3. **Interpreting Training Results**
   ```
   LunarLander-v2 example:
   - Simulation: ~50 Hz (Box2D default)
   - Episode: 200-300 steps
   - Real time: 4-6 seconds of simulated landing
   ```

**Practical Guide for New Environments:**

```python
# Step 1: Run a few random episodes to measure
env = gym.make("YourEnv-v0")
episode_lengths = []
for _ in range(10):
    obs = env.reset()
    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()  # Random
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        done = done or truncated
    episode_lengths.append(steps)

avg_length = np.mean(episode_lengths)
print(f"Average episode length: {avg_length}")

# Step 2: Set n_steps accordingly
recommended_n_steps = int(avg_length * 3)  # 3× average
print(f"Recommended n_steps: {recommended_n_steps}")

# Step 3: Choose nearest power of 2 for efficiency
n_steps = 2 ** round(np.log2(recommended_n_steps))
print(f"Use n_steps: {n_steps}")
```

### Adjusting Parameters for Different Episode Lengths

**Short Episodes (< 100 steps):**
```python
# Example: CartPole (episodes ~200 steps)
env = make_vec_env("CartPole-v1", n_envs=8)
model = PPO(
    env=env,
    n_steps=512,        # 2-3× episode length
    batch_size=64,
    n_epochs=4,
    # ... other params
)
```

**Medium Episodes (100-500 steps):**
```python
# Example: LunarLander (episodes ~250 steps)
env = make_vec_env("LunarLander-v2", n_envs=16)
model = PPO(
    env=env,
    n_steps=1024,       # 4× episode length ✓ Current setup
    batch_size=64,
    n_epochs=4,
)
```

**Long Episodes (> 1000 steps):**
```python
# Example: Atari or MuJoCo (episodes ~1000+ steps)
env = make_vec_env("HalfCheetah-v4", n_envs=4)
model = PPO(
    env=env,
    n_steps=2048,       # 2-3× episode length
    batch_size=64,
    n_epochs=10,        # Can reuse more with longer rollouts
)
```

**Very Long Episodes (> 5000 steps):**
```python
# Example: Long-horizon robotics tasks
env = make_vec_env("CustomRobot-v0", n_envs=2)
model = PPO(
    env=env,
    n_steps=4096,       # Still reasonable size
    batch_size=64,
    n_epochs=10,
    # Consider using fewer envs due to memory
)
```

### Balancing Parameters

**The magic formula:**
```
total_transitions = n_steps × num_envs
num_gradient_updates = (total_transitions / batch_size) × n_epochs

Typical target: 512 - 2048 gradient updates per learning phase
```

**If you change num_envs, adjust n_steps to keep total_transitions similar:**

```python
# Option 1: 16 envs
n_envs=16, n_steps=1024  → 16,384 transitions

# Option 2: 32 envs (faster collection)
n_envs=32, n_steps=512   → 16,384 transitions (same!)

# Both will have similar training behavior
# but Option 2 collects data faster (if you have enough CPU cores)
```

### Quick Reference Table

| Your Task | Episode Length | n_steps | num_envs | batch_size | n_epochs |
|-----------|---------------|---------|----------|------------|----------|
| Simple/Fast | 50-200 | 512 | 8-16 | 64 | 4 |
| Medium | 200-500 | 1024-2048 | 4-16 | 64 | 4-10 |
| Long | 500-2000 | 2048-4096 | 2-8 | 64 | 10 |
| Very Long | 2000+ | 4096+ | 1-4 | 64-128 | 10-20 |

**Memory constraints:** Larger n_steps × num_envs needs more RAM. Monitor with `htop`!

## 🧪 Experiments to Try

### Experiment 1: Compare Different Learning Rates
**Goal:** Understand how learning rate affects convergence speed.

Try these values in PPO:
```python
learning_rate=3e-4  # Default (current)
learning_rate=1e-3  # Faster learning
learning_rate=1e-4  # Slower learning
```

**Watch in TensorBoard:** Does higher learning rate learn faster but become unstable?

---

### Experiment 2: Exploration vs Exploitation
**Goal:** See how entropy coefficient affects exploration.

```python
ent_coef=0.01   # Current value
ent_coef=0.0    # No exploration bonus (pure exploitation)
ent_coef=0.1    # More exploration
```

**Watch:** `train/entropy_loss` in TensorBoard. Higher = more random actions.

---

### Experiment 3: Discount Factor (Gamma)
**Goal:** Understand how the agent values future rewards.

```python
gamma=0.999  # Current (values long-term rewards)
gamma=0.99   # Medium-term focus
gamma=0.9    # Short-term focus
```

**In LunarLander:** High gamma is important because landing safely requires planning ahead.

---

### Experiment 4: Training Duration
**Goal:** See when the agent stops improving.

Try different timesteps:
```python
total_timesteps=50000    # Quick test
total_timesteps=100000   # Current
total_timesteps=500000   # Better performance
total_timesteps=1000000  # Near-optimal
```

**Watch in TensorBoard:** When does `ep_rew_mean` plateau?

---

### Experiment 5: Number of Parallel Environments
**Goal:** Understand speed vs stability trade-off.

```python
env = make_vec_env("LunarLander-v2", n_envs=4)   # Slower, more stable
env = make_vec_env("LunarLander-v2", n_envs=16)  # Current
env = make_vec_env("LunarLander-v2", n_envs=32)  # Faster, less stable
```

**Trade-off:** More envs = faster wall-clock time, but less stable gradients.

---

## 📊 How to Analyze Results

### 1. **Check TensorBoard Graphs**
- Is `ep_rew_mean` increasing? → Agent is learning
- Is loss decreasing? → Policy is converging
- Is entropy decreasing? → Agent becoming more confident

### 2. **Compare Evaluation Results**
```bash
# Load and plot evaluations
import numpy as np
data = np.load('Solutions/Unit1/eval_logs/evaluations.npz')
print(data['results'])  # Rewards over time
print(data['timesteps'])  # When evaluated
```

### 3. **Test Different Checkpoints**
```python
# Load a specific checkpoint
model = PPO.load("Solutions/Unit1/checkpoints/ppo_lunarlander_10000_steps")
# Compare to best model
best_model = PPO.load("Solutions/Unit1/best_model/best_model")
```

---

## 🎓 Key Concepts to Learn

### PPO Hyperparameters Explained:

| Parameter | What it does | Typical range |
|-----------|-------------|---------------|
| `n_steps` | Steps to collect before update | 128-2048 |
| `batch_size` | Minibatch size for updates | 32-256 |
| `n_epochs` | How many times to reuse data | 3-10 |
| `gamma` | Discount factor for future rewards | 0.9-0.999 |
| `gae_lambda` | Advantage estimation smoothing | 0.9-0.99 |
| `ent_coef` | Exploration bonus | 0.0-0.1 |
| `learning_rate` | Step size for gradient descent | 1e-5 to 1e-3 |

### What to Look For:

1. **Good training:**
   - Steady increase in rewards
   - Smooth learning curves
   - Low variance between episodes

2. **Bad signs:**
   - Rewards plateauing early
   - Highly unstable/noisy rewards
   - Loss increasing instead of decreasing

---

## 🚀 Next Steps

1. **Run the enhanced training:**
   ```bash
   python Solutions/Unit1/Unit1.py
   ```

2. **Open TensorBoard:**
   ```bash
   tensorboard --logdir Solutions/Unit1/tensorboard_logs
   ```

3. **Try one experiment at a time** - change one hyperparameter, observe results

4. **Compare runs** - TensorBoard lets you overlay multiple runs to compare

5. **Visualize best model:**
   ```bash
   # Update eval_model.py to load best model:
   model_path = os.path.join(script_dir, "best_model", "best_model")
   ```

---

## 📝 Additional Resources

- [Stable-Baselines3 Callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)
- [TensorBoard Tutorial](https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html)
- [PPO Algorithm Explanation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Hyperparameter Tuning Guide](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
