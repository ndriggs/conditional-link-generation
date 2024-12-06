from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from link_generation.models.feature_extractor import ObsBraidFeaturesExtractor
from link_generation.envs.sig_det_env import SigDetEnv
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_rep', type=str, default='braid')
    parser.add_argument('--reward_type', type=str, default='sparse')
    parser.add_argument('--braid_index', type=int, default=7)
    parser.add_argument('--max_braid_length', type=int, default=60)
    parser.add_argument('--w1', type=float, default=0.7)
    parser.add_argument('--braid_or_knot_graph', type=str, default='braid')
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    return parser.parse_args()

def get_exp_name(args) :
    if args.state_rep == 'braid' :
        return f'braid_{args.reward_type}_{args.braid_index}_{args.w1}_{args.braid_or_knot_graph}_{args.seed}_{args.num_heads}_{args.num_layers}_{args.hidden_channels}'
    elif args.state_rep == 'ohe' :
        return f'ohe_{args.reward_type}_{args.braid_index}_{args.w1}_{args.seed}'
    
class LogSigAndLogDet(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.sigs = []
        self.log_dets = []

    def _on_step(self):
        for info in self.locals['infos'] :
            signature = info.get('signature', None)
            determinant = info.get('determinant', None)
            is_knot = info.get('is_knot', None)
            if (signature is not None) and (determinant is not None):
                # Log all variables to TensorBoard
                self.logger.record("abs_signature", np.abs(signature))
                self.logger.record("log_determinant", np.log1p(determinant))
                self.logger.record("is_knot", is_knot)
                self.sigs.append(np.abs(signature))
                self.log_dets.append(np.log1p(determinant))
        
        return True

def main() :
    args = parse_args()

    vec_env = make_vec_env("SigDetEnv-v0", n_envs=1, 
                           env_kwargs={'state_rep': args.state_rep,
                                       'reward_type': args.reward_type,
                                       'braid_index': args.braid_index,
                                       'max_braid_length': args.max_braid_length,
                                       'w1': args.w1,
                                       'seed': args.seed})
    
    if args.state_rep == 'braid' : 
        features_dim = (2**(args.num_layers-1))*args.hidden_channels*args.num_heads
        policy_kwargs = dict(
            features_extractor_class=ObsBraidFeaturesExtractor,
            features_extractor_kwargs=dict(num_heads=args.num_heads,
                                           num_layers=args.num_layers,
                                           hidden_channels=args.hidden_channels,
                                           braid_or_knot_graph=args.braid_or_knot_graph,
                                           braid_index=args.braid_index),
            net_arch=dict(pi=[2*features_dim, 2*features_dim], vf=[2*features_dim, 2*features_dim])
        )
        model = PPO("MlpPolicy", vec_env, n_steps=256, policy_kwargs=policy_kwargs, verbose=1, 
                    tensorboard_log='src/link_generation/train/logs/')
    elif args.state_rep == 'ohe' :
        features_dim = 2*(args.braid_index-1)*args.max_braid_length
        policy_kwargs = dict(
            net_arch=dict(pi=[2*features_dim, 2*features_dim], vf=[2*features_dim, 2*features_dim])
        )
        model = PPO("MlpPolicy", vec_env, n_steps=256, verbose=1, 
                    tensorboard_log='src/link_generation/train/logs/')
    training_steps = 30000 if args.reward_type == 'dense' else 150000
    model.learn(training_steps, callback=LogSigAndLogDet(), tb_log_name=get_exp_name(args))
    model.save('src/link_generation/train/models/'+get_exp_name(args))

if __name__ == '__main__' :
    main()
