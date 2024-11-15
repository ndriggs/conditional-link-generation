from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from link_generation.models.feature_extractor import BraidFeaturesExtractor
from link_generation.envs.signature_env import SignatureEnv
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_type', type=str, default='sparse')
    parser.add_argument('--cirriculum', type=lambda x: x.lower() == 'true', default=True)
    # how many features should the GNN extracts from the braid and target signature
    parser.add_argument('--features_dim', type=int, default=64)
    # whether to use the full training distribution or full test distribution when evaluating 
    parser.add_argument('--test', type=lambda x: x.lower() == 'true', default=True)
    # random seed to use 
    parser.add_argument('--seed', type=int, default=15)
    return parser.parse_args()

def main() :
    args = parse_args()
    vec_env = make_vec_env("SignatureEnv-v0", n_envs=4, 
                        env_kwargs={'reward_type': args.reward_type,
                                    'seed': args.seed,
                                    'cirriculum': args.cirriculum,
                                    'test': False})

    test_vec_env = make_vec_env("SignatureEnv-v0", n_envs=4, 
                                env_kwargs={'reward_type': 'sparse',
                                            'seed': args.seed,
                                            'test': args.test})


    policy_kwargs = dict(
        features_extractor_class=BraidFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
    )

    model = PPO("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

    mean_rewards, std_rewards = [], []
    for _ in range(250) :
        mean_reward, std_reward = evaluate_policy(model,test_vec_env,n_eval_episodes=200)
        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)
        print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        model.learn(total_timesteps=100)
    model.save("ppo_link_builder")
    np.save('PPO_mean_reward.npy', np.array(mean_rewards))
    np.save('PPO_std_reward.npy', np.array(std_rewards))


if __name__ == '__main__':
    main()