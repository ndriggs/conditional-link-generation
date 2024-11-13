import gymnasium as gym
from gymnasium import spaces
from typing import Any
from sage.all import BraidGroup, Link, Integer
from link_generation.models import GNN
import numpy as np

class SignatureEnv(gym.Env):

    metadata = {"render_modes": ["knot_diagram", "braid_word"], "render_fps": 2}

    def __init__(self, reward_type:str, braid_index: int = 7, cirriculum: bool = False,
                 curiousity: bool = False, render_mode:str ='knot_diagram'):
        '''
        params:
        reward_type: either 'dense' which will give the agent a reward everytime the signature
                     changes or 'sparse' which will only give the agent a reward when it finishes
                     the episode
        braid_index: the number of strands to braid
        cirriculum: whether to blinding train on all the training signatures or start with easy ones
                    and slowly move to more difficult signatures
        curiousity: if true then a model is trained to predict the signature of the new knot and an 
                    intrinsic reward is given for the difference between the predicted and actual signature.
                    Encourages the agent to explore new parts of the state space.
        render_mode: not currently working
        '''
        super(SignatureEnv, self).__init__()

        if braid_index < 3 : 
            raise ValueError(f"Invalid param: {braid_index}. 'braid_index' parameter must be greater than 2")

        if reward_type not in ['dense', 'sparse'] :
            raise ValueError(f"Invalid param: {reward_type}. 'reward_type' parameter must be one of 'dense' or 'sparse'")

        self.braid_index = braid_index 
        self.B = BraidGroup(self.braid_index)
        self.max_braid_length = 50 # somewhat arbitrary, still computes signature for longer braids
        self.reward_type = reward_type
        self.num_envs = 1 # so StableBaselines3 can use VecEnv
        self.render_mode = render_mode
        self.curiousity = curiousity
        self.cirriculum = cirriculum
        self.episode_num = 0
        
        if self.cirriculum : # start off easy
            self.train_target_signatures = [-2, -1, 1, 2]
        else : # train is +- 1,2,5,6,9,10,...,37,38
            self.train_target_signatures = [
                sig for sublist in [(-2*i,-2*i+1,2*i-1,2*i) for i in range(1,(self.max_braid_length-10)/2,2)] for sig in sublist
            ]
        # test is +- 3,4,7,8,11,12,...,35,36
        self.test_target_signatures = [
            sig for sublist in [(-2*i,-2*i+1,2*i-1,2*i) for i in range(2,(self.max_braid_length-10)/2,2)] for sig in sublist
        ]

        # braid_index = 3 would give 5 actions: {sigma_1, sigma_2, sigma_1^{-1}, sigma_2^{-1}, STOP}
        # 0 represents the STOP action, all others are what you'd expect
        self.action_space = spaces.Discrete((self.braid_index-1)*2 + 1, start=-self.braid_index+1) 

        # create the observation space
        signature_space = spaces.Box(low=-self.max_braid_length, high=self.max_braid_length, shape=(1,), dtype=np.int32)
        self.observation_space = spaces.Dict({
            'observation': spaces.Sequence(spaces.Discrete((self.braid_index-1)*2 + 1, start=-self.braid_index+1)),
            'achieved_goal': signature_space,
            'desired_goal': signature_space
        })

        if self.curiousity :
            self.model = GNN(hidden_channels=32, num_heads=8, num_layers=4, dropout=0,
                             classification=False, both=True, ohe_inverses=True, 
                             double_features=True)
            

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None): # target_signature: int
        if seed is not None :
            np.random.seed(seed)
        
        # initialize the braid word with a random generator or inverse
        self.braid_word = [Integer(np.random.choice([i for i in range(-self.braid_index+1, self.braid_index)  if i != 0]))]
        self.link = Link(self.B(self.braid_word))
        self.current_signature = self.link.signature()
        self.t_minus_1_signature = self.current_signature
        self.episode_num += 1

        if self.cirriculum and (self.episode_num % 20 == 0) and (self.episode_num // 20 <= 9):
            # add four more slightly farther away signatures every 20 episodes, up to a certain point
            self.train_target_signatures = [
                sig for sublist in [(-2*i,-2*i+1,2*i-1,2*i) for i in range(1,((self.episode_num//20)+1)*2,2)] for sig in sublist
            ]

        if options is not None and options.get('test', False):
            self.target_signature = np.random.choice(self.test_target_signatures)
        else:
            self.target_signature = np.random.choice(self.train_target_signatures)

        observation = {
            'observation': self.braid_word,
            'achieved_goal': self.current_signature,
            'desired_goal': self.target_signature
        }

        return observation, {}


    def step(self, action):
        # the STOP token was selected
        if action == 0 :
            terminated = True
            truncated = False

            # calculate the reward
            if self.reward_type == 'dense' :
                reward = 0
            elif self.reward_type == 'sparse' :
                distance_from_goal = np.abs(self.current_signature - self.target_signature)
                reward = 1 / (1 + distance_from_goal)
        
        else : 
            self.braid_word.append(Integer(action))
            self.link = Link(self.B(self.braid_word))
            self.current_signature = self.link.signature()

            terminated = False 
            if len(self.braid_word) >= self.max_braid_length :
                truncated = True
            else :
                truncated = False

            # calculate the reward
            if self.reward_type == 'dense' :
                reward = np.abs(self.t_minus_1_signature - self.target_signature) \
                    - np.abs(self.current_signature - self.target_signature)
            elif self.reward_type == 'sparse' :
                reward = 0

            self.t_minus_1_signature = self.current_signature

        observation = {
            'observation': self.braid_word,
            'achieved_goal': self.current_signature,
            'desired_goal': self.target_signature
        }

        return observation, reward, terminated, truncated, {}

    def render(self):
        self.link.plot()

    def close(self):
        pass

    
