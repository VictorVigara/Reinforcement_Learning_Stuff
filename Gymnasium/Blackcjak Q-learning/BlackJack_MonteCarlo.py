from collections import defaultdict
import gymnasium as gym
import numpy as np

class BlackjackMCAgent:
    def __init__(
            self, 
            env: gym.Env, 
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
    ):
        """Initialize a Monte Carlo agent.

        Args:
            env: The training environment
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
        """

        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns = defaultdict(list)  # Store returns for each (state, action)

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track episode data
        self.episode_data = []  # [(state, action), ...]
        self.episode_rewards = []  # Track rewards for plotting

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Random action

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def store_transition(self, obs, action):
        """Store state-action pairs during episode"""
        self.episode_data.append((obs, action))
    
    def update_episode(self, episode_reward):
        """Update Q-values after complete episode using actual return"""
        # In Blackjack, all actions get the same final reward
        # (win/lose/draw applies to the entire hand)
        for obs, action in self.episode_data:
            # Store the actual return
            self.returns[(obs, action)].append(episode_reward)
            # Update Q-value to average of all returns seen for this (state, action)
            self.q_values[obs][action] = np.mean(self.returns[(obs, action)])
        
        # Track episode reward for plotting
        self.episode_rewards.append(episode_reward)
        
        # CRITICAL: Clear episode data for next episode
        self.episode_data = []

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# Training hyperparameters
n_episodes = 100000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration

# Create environment and agent
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackMCAgent(
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

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Episode rewards over time (instead of training error)
axs[2].set_title("Episode Rewards Over Time")
episode_reward_moving_average = get_moving_avgs(
    agent.episode_rewards,
    rolling_length,
    "same"
)
axs[2].plot(range(len(episode_reward_moving_average)), episode_reward_moving_average)
axs[2].set_ylabel("Episode Reward")
axs[2].set_xlabel("Episode")

plt.tight_layout()
plt.show()


# Test the trained agent
def test_agent(agent, env, num_episodes=1000):
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