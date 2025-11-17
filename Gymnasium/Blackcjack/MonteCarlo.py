import gymnasium as gym
import sys
import os

# Add the MonteCarlo directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MonteCarlo'))

from MonteCarlo_Agent import MonteCarloAgent
from MonteCarlo_Training import train_monte_carlo, test_agent, plot_training_results

# Training hyperparameters
n_episodes = 100000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration

# Create environment
env = gym.make("Blackjack-v1", sab=False)

agent = MonteCarloAgent(
    env=env,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Training using common function
print("Training Monte Carlo agent...")
agent, stats = train_monte_carlo(
    env=env,
    n_episodes=n_episodes,
    initial_epsilon=start_epsilon,
    final_epsilon=final_epsilon,
    plot_results=False,  # We'll plot separately
    test_episodes=1000,  # Fix: was 0, causing NaN errors
    rolling_window=500,
    verbose=True
)

# Get the wrapped environment from training for plotting
wrapped_env = stats.get('wrapped_env', env)

# Plot training results using common function
print("\nPlotting training results...")
try:
    plot_training_results(wrapped_env, agent, rolling_window=500)
except AttributeError:
    print("Note: Plotting requires environment statistics. Training completed successfully but plots unavailable.")
    print("You can enable built-in plotting by setting plot_results=True in train_monte_carlo()")

# Print final statistics (test results already available from train_monte_carlo)
print("\n" + "="*50)
print("FINAL BLACKJACK RESULTS")
print("="*50)
test_results = stats['test_results']
print(f"Win Rate: {test_results['success_rate']:.1%}")
print(f"Average Reward: {test_results['average_reward']:.3f}")
print(f"Standard Deviation: {test_results['std_reward']:.3f}")
print(f"States learned: {len(agent.q_values)}")
print(f"State-action pairs: {len(agent.returns)}")