import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
import sys
import os

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (rl-link-builder)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

register(
    id='LinkBuilderEnv-v0',  
    entry_point='envs.signature_environment:LinkBuilderEnv', 
)

# Make a vectorized environment
vec_env = make_vec_env("LinkBuilderEnv-v0", n_envs=4, 
                       env_kwargs={'reward_type': 'sparse'})

# Normalize the vectorized environment
vec_env = VecNormalize(vec_env, clip_obs=6e13, clip_reward=100, gamma=1)

model = PPO("MlpPolicy",vec_env,verbose=1)

mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=200, warn=False)
print(f"mean_reward before training: {mean_reward:.2f} +/- {std_reward:.2f}")

model.learn(total_timesteps=29000)
model.save("ppo_link_builder")
vec_env.save("normalized_vec_env.pkl")

mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=200)
print(f"mean_reward after training:{mean_reward:.2f} +/- {std_reward:.2f}")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_link_builder")

# rewards_per_episode = []
# for episode in range(100) : 
#     rewards_this_episode = []
#     obs = env.reset()
#     done = False 
#     while not done : 
#         action, _states = model.predict(obs)
#         obs, reward, terminated, truncated, info = env.step(action)
#         rewards_this_episode.append(reward)
#         done = terminated or truncated
#     rewards_per_episode.append(sum(rewards_this_episode))

# print('mean:', np.mean(rewards_per_episode))
# print('std:', np.std(rewards_per_episode))

# plt.figure(figsize=(10, 6))
# plt.plot(rewards_per_episode)
# plt.ylabel('Reward per Episode')
# plt.xlabel('Episode Number')
# plt.savefig('cumulative_reward.png')