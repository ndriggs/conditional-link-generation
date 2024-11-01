from stable_baselines3 import HerReplayBuffer, DQN
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from scipy.stats import ttest_ind
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
    runtime_warnings = []
    obs = env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        for i, done in enumerate(dones) :
            if done : 
                if 'time_taken' in infos[i].keys() :
                    time_taken.append(infos[i]['time_taken'])
                elif 'missed_target' in infos[i].keys() :
                    missed_targets.append(infos[i]['missed_target'])
                if infos[i]['RuntimeWarning'] :
                    runtime_warnings.append([infos[i]['braid_length'], infos[i]['signature']])
                    print('RuntimeWarning, braid length', infos[i]['braid_length'])
                    print('Signature', infos[i]['signature'])
    print('percent goals achieved: ', len(time_taken)/(len(time_taken)+len(missed_targets)))
    time_taken = np.array(time_taken)
    print('time taken avg: ', np.mean(time_taken), 'std', np.std(time_taken), 'max', np.max(time_taken), 'min', np.min(time_taken))
    print('missed targets avg:', np.mean(np.abs(np.array(missed_targets))), 'min', np.min(np.abs(np.array(missed_targets))))
    return time_taken, np.array(missed_targets)



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

mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=200, warn=False)
print(f"mean_reward before training: {mean_reward:.2f} +/- {std_reward:.2f}")

print('pre training evaluation')
pre_time_taken, pre_missed_targets = evaluate_vec_env(vec_env, model)

# Train the model
model.learn(40000)

print('post training evaluation')
post_time_taken, post_missed_targets = evaluate_vec_env(vec_env, model)

print('time taken')
print(ttest_ind(pre_time_taken, post_time_taken, equal_var=False))

print('missed targets')
print(ttest_ind(pre_missed_targets, post_missed_targets, equal_var=False))

mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=200)
print(f"mean_reward after training: {mean_reward:.2f} +/- {std_reward:.2f}")

np.save('pre_time_taken.npy', pre_time_taken)
np.save('post_time_taken.npy', post_time_taken)
np.save('pre_missed_targets.npy', pre_missed_targets)
np.save('post_missed_targets.npy', post_missed_targets)

model.save("her_env")
vec_env.save("normalized_vec_env.pkl")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
# model = model_class.load("her_env", env=env)



