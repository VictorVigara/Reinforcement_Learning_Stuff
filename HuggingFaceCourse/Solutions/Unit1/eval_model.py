from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import os

# Load model from the same folder as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best_model/best_model.zip")

env = Monitor(gym.make("LunarLander-v2", render_mode='human'))
model = PPO.load(path=model_path, env=env)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
