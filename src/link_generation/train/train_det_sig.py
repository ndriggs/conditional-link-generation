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
    parser.add_argument('--w1', type=float, default=0.7)
    parser.add_argument('--braid_or_knot_graph', type=str, default='braid')
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    return parser.parse_args()

def get_exp_name(args) :
    if args.state_rep == 'braid' :
        return f'braid_{args.reward_type}_{args.w1}_{args.braid_or_knot_graph}_{args.seed}_{args.num_heads}_{args.num_layers}_{args.hidden_channels}'
    elif args.state_rep == 'ohe' :
        return f'ohe_{args.reward_type}_{args.w1}_{args.seed}'
    
class LogSigAndLogDet(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.sigs = []
        self.log_dets = []

    def _on_step(self):
        env = self.training_env.envs[0]
        
        # Check if the episode has ended
        if self.training_env.get_attr("_episode_ended")[0]:
            sig = env.current_signature
            log_det = np.log1p(env.current_det)
            self.sigs.append(sig)
            self.dets.append(log_det)
            
            # Log all variables to TensorBoard
            self.logger.record("signature", sig)
            self.logger.record("log_determinant", log_det)
        
        return True

def main() :
    args = parse_args()

    vec_env = make_vec_env("SigDetEnv-v0", n_envs=4, 
                           env_kwargs={'state_rep': args.state_rep,
                                       'reward_type': args.reward_type,
                                       'seed': args.seed})
    
    if args.state_rep == 'braid' : 
        policy_kwargs = dict(
            features_extractor_class=ObsBraidFeaturesExtractor,
            features_extractor_kwargs=dict(num_heads=args.num_heads,
                                           num_layers=args.num_layers,
                                           hidden_channels=args.hidden_channels,
                                           braid_or_knot_graph=args.braid_or_knot_graph),
        )
        model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1, 
                    tensorboard_log='src/link_generation/train/logs/')
    elif args.state_rep == 'ohe' :
        model = PPO("MlpPolicy", vec_env, verbose=1, 
                    tensorboard_log='src/link_generation/train/logs/')
    model.learn(5000, callback=LogSigAndLogDet(), tb_log_name=get_exp_name(args))

if __name__ == '__main__' :
    main()
