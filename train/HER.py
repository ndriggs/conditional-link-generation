from stable_baselines3 import HerReplayBuffer, DQN
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import gymnasium as gym
import os
import sys

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (rl-link-builder)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from envs.signature_goal_env import LinkBuilderEnv


def evaluate(env, model) :
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



# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

env = LinkBuilderEnv()
model = DQN(
    "MultiInputPolicy",
    env,
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
evaluate(env, model)

# Train the model
model.learn(2000)

print('post training evaluation')
evaluate(env, model)

model.save("her_env")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
# model = model_class.load("her_env", env=env)



