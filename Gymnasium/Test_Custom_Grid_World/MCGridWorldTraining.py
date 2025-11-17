import sys
import os

import gymnasium as gym
import gymnasium_env

import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MonteCarlo.MonteCarlo_Agent import MonteCarloAgent

# Import the hashable wrapper (put this file in the same directory)
from gymnasium_env.wrappers import HashableObservationWrapper

def main():
    """Main training function."""
    # Training hyperparameters
    n_episodes = 20000        # Number of episodes to practice (reduced for testing)
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.01         # Always keep some exploration

    # Create environment and agent
    env = gym.make('gymnasium_env/GridWorld-v0')
    # Store reference to RecordEpisodeStatistics wrapper for plotting
    stats_env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    env = HashableObservationWrapper(stats_env)

    agent = MonteCarloAgent(
        env=env,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    from tqdm import tqdm  # Progress bar
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(obs)
            
            # Store this transition
            agent.store_transition(obs, action)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            done = terminated or truncated
            obs = next_obs

        # Update Q-values using the complete episode return
        agent.update_episode(episode_reward)
        agent.decay_epsilon()

    from matplotlib import pyplot as plt

    def get_moving_avgs(arr, window, convolution_mode):
        """Compute moving average to smooth noisy data."""
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    # Smooth over a smaller window for fewer episodes
    rolling_length = min(100, n_episodes // 10)  # Adaptive window size
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance) - access from stats_env
    axs[0].set_title("Episode rewards")
    if len(stats_env.return_queue) > rolling_length:
        reward_moving_average = get_moving_avgs(
            stats_env.return_queue,
            rolling_length,
            "valid"
        )
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    else:
        axs[0].plot(stats_env.return_queue)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per episode)
    axs[1].set_title("Episode lengths")
    if len(stats_env.length_queue) > rolling_length:
        length_moving_average = get_moving_avgs(
            stats_env.length_queue,
            rolling_length,
            "valid"
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
    else:
        axs[1].plot(stats_env.length_queue)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Episode rewards over time (from agent)
    axs[2].set_title("Episode Rewards Over Time")
    if len(agent.episode_rewards) > rolling_length:
        episode_reward_moving_average = get_moving_avgs(
            agent.episode_rewards,
            rolling_length,
            "same"
        )
        axs[2].plot(range(len(episode_reward_moving_average)), episode_reward_moving_average)
    else:
        axs[2].plot(agent.episode_rewards)
    axs[2].set_ylabel("Episode Reward")
    axs[2].set_xlabel("Episode")

    plt.tight_layout()
    plt.show()

    # Test the trained agent
    def test_agent(agent, env, num_episodes=100):  # Reduced for quicker testing
        """Test agent performance without learning or exploration."""
        total_rewards = []

        # Temporarily disable exploration for testing
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # Pure exploitation

        for _ in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)

        # Restore original epsilon
        agent.epsilon = old_epsilon

        win_rate = np.mean(np.array(total_rewards) > 0)
        average_reward = np.mean(total_rewards)

        print(f"Test Results over {num_episodes} episodes:")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Average Reward: {average_reward:.3f}")
        print(f"Standard Deviation: {np.std(total_rewards):.3f}")

    # Test your agent
    test_agent(agent, env)


if __name__ == "__main__":
    main()