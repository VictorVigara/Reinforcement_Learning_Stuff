from collections import defaultdict
import numpy as np

class MonteCarloAgent:
    """Generic MC agent that works with any discrete environment"""
    def __init__(self, env, initial_epsilon, epsilon_decay, final_epsilon):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns = defaultdict(list)
        
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.episode_data = []
        self.episode_rewards = []

    def get_action(self, obs):
        # Works for any hashable observation
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
    
    def store_transition(self, obs, action):
        self.episode_data.append((obs, action))
    
    def update_episode(self, episode_reward):
        for obs, action in self.episode_data:
            self.returns[(obs, action)].append(episode_reward)
            self.q_values[obs][action] = np.mean(self.returns[(obs, action)])
        
        self.episode_rewards.append(episode_reward)
        self.episode_data = []

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)