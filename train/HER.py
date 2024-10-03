from stable_baselines3 import HerReplayBuffer, DQN
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import os
import sys

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (rl-link-builder)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from envs.signature_goal_env import LinkBuilderEnv


def evaluate_single_env(env, model) :
    time_taken = []
    missed_targets = []
    obs, info = env.reset()
    for _ in range(2500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated :
            time_taken.append(info['time_taken'])
            obs, info = env.reset()
        elif truncated :
            missed_targets.append(info['missed_target'])
    print('percent goals achieved: ', len(time_taken)/(len(time_taken)+len(missed_targets)))
    time_taken = np.array(time_taken)
    print('time taken avg: ', np.mean(time_taken), 'std', np.std(time_taken), 'max', np.max(time_taken), 'min', np.min(time_taken))
    print('missed targets avg:', np.mean(np.abs(np.array(missed_targets))), 'min', np.min(np.abs(np.array(missed_targets))))

def evaluate_vec_env(env, model) :
    time_taken = []
    missed_targets = []
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        for i, done in enumerate(dones) :
            if done : 
                if 'time_taken' in infos[i].keys() :
                    time_taken.append(infos[i]['time_taken'])
                elif 'missed_target' in infos[i].keys() :
                    missed_targets.append(infos[i]['missed_target'])
    print('percent goals achieved: ', len(time_taken)/(len(time_taken)+len(missed_targets)))
    time_taken = np.array(time_taken)
    print('time taken avg: ', np.mean(time_taken), 'std', np.std(time_taken), 'max', np.max(time_taken), 'min', np.min(time_taken))
    print('missed targets avg:', np.mean(np.abs(np.array(missed_targets))), 'min', np.min(np.abs(np.array(missed_targets))))



# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

register(
    id='LinkBuilderEnv-v1',  
    entry_point='envs.signature_goal_env:LinkBuilderEnv', 
)

# Make a vectorized environment
vec_env = make_vec_env("LinkBuilderEnv-v1", n_envs=4)

# Normalize the vectorized environment
vec_env = VecNormalize(vec_env, norm_reward=False, clip_obs=6e13, clip_reward=100, gamma=0.97)
model = DQN(
    "MultiInputPolicy",
    vec_env,
    replay_buffer_class=HerReplayBuffer,
    gamma=.97,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
    ),
    verbose=1,
)

print('pre training evaluation')
evaluate_vec_env(vec_env, model)

# Train the model
model.learn(20000)

print('post training evaluation')
evaluate(vec_env, model)

model.save("her_env")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
# model = model_class.load("her_env", env=env)



