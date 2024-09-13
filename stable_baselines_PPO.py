import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

from gymnasium.envs.registration import register

register(
    id='LinkBuilderEnv-v0',  
    entry_point='envs.signature_environment:LinkBuilderEnv',  
)

# Parallel environments
env = gym.make("LinkBuilderEnv-v0")

model = PPO("MlpPolicy",env,verbose=1)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)
print(f"mean_reward before training: {mean_reward:.2f} +/- {std_reward:.2f}")

model.learn(total_timesteps=25000)
model.save("ppo_link_builder")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
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