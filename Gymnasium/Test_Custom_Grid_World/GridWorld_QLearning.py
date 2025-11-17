import sys
import os

from collections import defaultdict
import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from QLearning.QLearningAgent import QLearningAgent
from gymnasium_env.wrappers import HashableObservationWrapper


# Training hyperparameters
learning_rate = 0.1        # How fast to learn (higher = faster but less stable)
n_episodes = 20000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration

# Create environment and agent
env = gym.make('gymnasium_env/GridWorld-v0')
# Store reference to RecordEpisodeStatistics wrapper for plotting
stats_env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
env = HashableObservationWrapper(stats_env)

agent = QLearningAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

from tqdm import tqdm  # Progress bar

for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset()
    done = False

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Learn from this experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Move to next state
        done = terminated or truncated
        obs = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()

from matplotlib import pyplot as plt

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance) - access from stats_env
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    stats_env.return_queue,  # Access from stats_env instead of env
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand) - access from stats_env
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    stats_env.length_queue,  # Access from stats_env instead of env
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()

# Test the trained agent with rendering
def test_agent(agent, env, num_episodes=5, render_episodes=100):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    # Create a separate environment for testing with rendering
    test_env = gym.make('gymnasium_env/GridWorld-v0', render_mode='human')
    test_env = HashableObservationWrapper(test_env)

    print(f"\nTesting agent for {num_episodes} episodes (rendering first {render_episodes})...")
    
    for episode in range(num_episodes):
        # Use rendered environment for first few episodes
        current_env = test_env if episode < render_episodes else env
        
        obs, info = current_env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        if episode < render_episodes:
            print(f"\n--- Episode {episode + 1} (with rendering) ---")
            current_env.render()
            input("Press Enter to start episode...")

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = current_env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated

            if episode < render_episodes:
                current_env.render()
                print(f"Step {step_count}: Action={action}, Reward={reward:.3f}, Done={done}")
                if not done:
                    input("Press Enter for next step...")

        total_rewards.append(episode_reward)
        
        if episode < render_episodes:
            print(f"Episode {episode + 1} finished: Total Reward = {episode_reward:.3f}")
            input("Press Enter to continue to next episode...")

    # Close the test environment
    test_env.close()
    
    # Restore original epsilon
    agent.epsilon = old_epsilon

    # Calculate statistics
    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"\n=== Test Results over {num_episodes} episodes ===")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")
    print(f"Min Reward: {np.min(total_rewards):.3f}")
    print(f"Max Reward: {np.max(total_rewards):.3f}")

# Test your agent
test_agent(agent, env)