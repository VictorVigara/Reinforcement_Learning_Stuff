import gymnasium as gym
import os

# from huggingface_sb3 import load_from_hub, package_to_hub
# from huggingface_hub import (
#     notebook_login,
# )  # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

# Get script directory for saving outputs
script_dir = os.path.dirname(os.path.abspath(__file__))

# First, we create our environment called LunarLander-v2
# env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
env = make_vec_env("LunarLander-v2", n_envs=16)

# Create separate evaluation environment for callbacks (must be single env, not vectorized)
eval_env = Monitor(gym.make("LunarLander-v2"))

# Define TensorBoard log directory
tensorboard_log_dir = os.path.join(script_dir, "tensorboard_logs")

# Define the model with TensorBoard logging
model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,           # Discount factor: how much to value future rewards
    gae_lambda = 0.98,       # GAE parameter for advantage estimation
    ent_coef = 0.01,         # Entropy coefficient: encourages exploration
    verbose=1,
    tensorboard_log=tensorboard_log_dir)  # Enable TensorBoard logging!

# Setup callbacks for monitoring during training
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # Save checkpoint every 10k steps
    save_path=os.path.join(script_dir, "checkpoints"),
    name_prefix="ppo_lunarlander"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(script_dir, "best_model"),
    log_path=os.path.join(script_dir, "eval_logs"),
    eval_freq=5000,        # Evaluate every 5k steps
    n_eval_episodes=10,    # Run 10 episodes for each evaluation
    deterministic=True,
    render=False
)

# Combine callbacks
callbacks = CallbackList([checkpoint_callback, eval_callback])

# Train the agent with callbacks
print("Starting training...")
print(f"TensorBoard logs: {tensorboard_log_dir}")
print("To view training in real-time, run in a separate terminal:")
print(f"  tensorboard --logdir {tensorboard_log_dir}")
print()

model.learn(total_timesteps=1000000, callback=callbacks)

# Save the final model
model_name = os.path.join(script_dir, "ppo-LunarLander-v2")
model.save(model_name)
print(f"Model saved to: {model_name}")

# Final evaluation
print("\nRunning final evaluation...")
eval_env_final = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env_final, n_eval_episodes=20, deterministic=True)
print(f"Final mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")