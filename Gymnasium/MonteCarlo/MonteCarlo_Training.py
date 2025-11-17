import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from MonteCarlo_Agent import MonteCarloAgent


def train_monte_carlo(
    env, 
    n_episodes=50000, 
    initial_epsilon=1.0, 
    final_epsilon=0.1, 
    plot_results=True,
    test_episodes=1000,
    rolling_window=500,
    verbose=True
):
    """
    Train a Monte Carlo agent on any discrete environment.
    
    Args:
        env: Gymnasium environment (should have discrete action space)
        n_episodes: Number of training episodes
        initial_epsilon: Starting exploration rate
        final_epsilon: Minimum exploration rate  
        plot_results: Whether to plot training curves
        test_episodes: Number of episodes for final evaluation
        rolling_window: Window size for moving averages in plots
        verbose: Whether to show progress bar and print results
    
    Returns:
        agent: Trained MonteCarloAgent
        training_stats: Dictionary with training statistics
    """
    
    # Calculate epsilon decay
    epsilon_decay = initial_epsilon / (n_episodes / 2)
    
    # Wrap environment to track statistics
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    
    # Create agent
    agent = MonteCarloAgent(
        env=env,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon
    )
    
    if verbose:
        print(f"Training Monte Carlo agent for {n_episodes} episodes...")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
    
    # Training loop
    iterator = tqdm(range(n_episodes)) if verbose else range(n_episodes)
    
    for episode in iterator:
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(obs)
            agent.store_transition(obs, action)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            done = terminated or truncated
            obs = next_obs

        # Update Q-values using complete episode return
        agent.update_episode(episode_reward)
        agent.decay_epsilon()
    
    # Test the trained agent
    test_results = test_agent(agent, env, test_episodes, verbose)
    
    # Plot training curves
    if plot_results:
        plot_training_results(env, agent, rolling_window)
    
    # Compile training statistics
    training_stats = {
        'final_epsilon': agent.epsilon,
        'total_states_visited': len(agent.q_values),
        'total_state_action_pairs': len(agent.returns),
        'test_results': test_results
    }
    
    if verbose:
        print(f"\nTraining completed!")
        print(f"Final epsilon: {agent.epsilon:.3f}")
        print(f"States discovered: {len(agent.q_values)}")
        print(f"State-action pairs learned: {len(agent.returns)}")
    
    return agent, training_stats


def test_agent(agent, env, num_episodes=1000, verbose=True):
    """Test agent performance without exploration"""
    total_rewards = []
    episode_lengths = []

    # Temporarily disable exploration
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    # Calculate statistics
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    # Success rate (positive reward episodes)
    success_rate = np.mean(np.array(total_rewards) > 0)

    test_results = {
        'average_reward': avg_reward,
        'std_reward': std_reward,
        'success_rate': success_rate,
        'average_episode_length': avg_length
    }

    if verbose:
        print(f"\nTest Results over {num_episodes} episodes:")
        print(f"Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Episode Length: {avg_length:.1f}")

    return test_results


def plot_training_results(env, agent, rolling_window=500):
    """Plot training curves"""
    def get_moving_average(arr, window):
        return np.convolve(arr, np.ones(window), 'valid') / window

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

    # Episode rewards
    axs[0].set_title("Episode Rewards")
    if len(env.return_queue) > rolling_window:
        reward_ma = get_moving_average(env.return_queue, rolling_window)
        axs[0].plot(range(len(reward_ma)), reward_ma)
    else:
        axs[0].plot(env.return_queue)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths
    axs[1].set_title("Episode Lengths")
    if len(env.length_queue) > rolling_window:
        length_ma = get_moving_average(env.length_queue, rolling_window)
        axs[1].plot(range(len(length_ma)), length_ma)
    else:
        axs[1].plot(env.length_queue)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Cumulative rewards
    axs[2].set_title("Agent Episode Rewards")
    if len(agent.episode_rewards) > rolling_window:
        agent_reward_ma = get_moving_average(agent.episode_rewards, rolling_window)
        axs[2].plot(range(len(agent_reward_ma)), agent_reward_ma)
    else:
        axs[2].plot(agent.episode_rewards)
    axs[2].set_ylabel("Episode Reward")
    axs[2].set_xlabel("Episode")

    plt.tight_layout()
    plt.show()


# Example usage functions for different environments
def train_blackjack(plot=True):
    """Train MC agent on Blackjack"""
    env = gym.make("Blackjack-v1", sab=False)
    agent, stats = train_monte_carlo(
        env, 
        n_episodes=100000,
        plot_results=plot
    )
    return agent, stats


def train_gridworld(env_id="gymnasium_env/GridWorld-v0", plot=True):
    """Train MC agent on GridWorld (requires custom environment)"""
    env = gym.make(env_id)
    # Add wrapper if needed for observation format
    from gymnasium.wrappers import FlattenObservation
    env = FlattenObservation(env)
    
    agent, stats = train_monte_carlo(
        env, 
        n_episodes=50000,
        plot_results=plot
    )
    return agent, stats


if __name__ == "__main__":
    # Example: Train on Blackjack
    print("Training Monte Carlo agent on Blackjack...")
    agent, stats = train_blackjack()
