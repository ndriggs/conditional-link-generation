from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from link_generation.models.feature_extractor import BraidFeaturesExtractor
from link_generation.envs.signature_env import SignatureEnv
import numpy as np

# Make a vectorized environment
vec_env = make_vec_env("SignatureEnv-v0", n_envs=4, 
                       env_kwargs={'reward_type': 'sparse'})

policy_kwargs = dict(
    features_extractor_class=BraidFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=64),
)

model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

def eval_policy(model, vec_env) :
    reset_options = {'test': True}
    mean_reward, std_reward = evaluate_policy(
        model,
        vec_env,
        n_eval_episodes=200,
        deterministic=True,
        render=False,
        callback=None,
        reward_threshold=None,
        return_episode_rewards=False,
        warn=True,
        reset_options=reset_options
    )
    return mean_reward, std_reward

mean_rewards, std_rewards = [], []
for _ in range(250) :
    mean_reward, std_reward = eval_policy(model, vec_env)
    mean_rewards.append(mean_reward)
    std_rewards.append(std_reward)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    model.learn(total_timesteps=100)
model.save("ppo_link_builder")
np.save('PPO_mean_reward.npy', np.array(mean_rewards))
np.save('PPO_std_reward.npy', np.array(std_rewards))
