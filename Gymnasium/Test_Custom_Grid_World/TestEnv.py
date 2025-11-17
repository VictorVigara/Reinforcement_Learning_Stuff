import gymnasium as gym
import gymnasium_env  # Import the custom environment module

# Create the environment like any built-in environment
env = gym.make("gymnasium_env/GridWorld-v0")
print(env)
print(env.unwrapped.size)
# <OrderEnforcing<PassiveEnvChecker<GridWorld<gymnasium_env/GridWorld-v0>>>>

# Customize environment parameters
env = gym.make("gymnasium_env/GridWorld-v0", size=10)
print(env.unwrapped.size)
# 10

print(env.observation_space)
# Dict('agent': Box(0, 9, (2,), int64), 'target': Box(0, 9, (2,), int64))
print(env.action_space)
# Discrete(4)
print(env.reset())
# {'agent': array([0, 0]), 'target': array([1, 1])}

# Create multiple environments for parallel training
vec_env = gym.make_vec("gymnasium_env/GridWorld-v0", num_envs=3)
print(vec_env)
# SyncVectorEnv(gymnasium_env/GridWorld-v0, num_envs=3)

from gymnasium.utils.env_checker import check_env

# This will catch many common issues
try:
    check_env(env.unwrapped)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")

# In our example, observations cannot be used directly in learning code because they are
# dictionaries. However, we don’t actually need to touch our environment implementation
# to fix this! We can simply add a wrapper on top of environment instances to flatten
# observations into a single array:

import gymnasium
import gymnasium_env
from gymnasium.wrappers import FlattenObservation

env = gymnasium.make('gymnasium_env/GridWorld-v0')
wrapped_env = FlattenObservation(env)
print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}



# Wrappers have the big advantage that they make environments highly modular.
# For instance, instead of flattening the observations from GridWorld, you might only
# want to look at the relative position of the target and the agent.
# In the section on ObservationWrappers we have implemented a wrapper that does
# this job. This wrapper is also available in gymnasium_env/wrappers/relative_position.py:

import gymnasium
import gymnasium_env
from gymnasium_env.wrappers import RelativePosition

env = gymnasium.make('gymnasium_env/GridWorld-v0')
wrapped_env = RelativePosition(env)
print(wrapped_env.reset())     # E.g.  [-3  3], {}