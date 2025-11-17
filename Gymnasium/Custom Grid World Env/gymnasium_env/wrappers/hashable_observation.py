import gymnasium as gym
import numpy as np
from gymnasium.spaces import Space


class HashableObservationWrapper(gym.ObservationWrapper):
    """Wrapper that converts observations to hashable format for tabular RL methods."""
    
    def __init__(self, env):
        super().__init__(env)
        # Keep the original observation space for reference
        self._original_observation_space = env.observation_space
        
    def observation(self, obs):
        """Convert observation to hashable format."""
        return self._make_hashable(obs)
    
    def _make_hashable(self, obs):
        """Convert any observation to a hashable tuple format."""
        if isinstance(obs, dict):
            # Convert dict with numpy arrays to tuple of tuples
            items = []
            for key in sorted(obs.keys()):  # Sort for consistent ordering
                value = obs[key]
                if isinstance(value, np.ndarray):
                    items.append((key, tuple(value.flatten())))
                else:
                    items.append((key, value))
            return tuple(items)
        elif isinstance(obs, np.ndarray):
            return tuple(obs.flatten())
        elif isinstance(obs, (list, tuple)):
            # Handle nested lists/tuples that might contain numpy arrays
            return tuple(self._make_hashable(item) if hasattr(item, '__iter__') and not isinstance(item, str)
                        else item for item in obs)
        else:
            # Already hashable (int, float, str, etc.)
            return obs