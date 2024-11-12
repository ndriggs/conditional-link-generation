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

        
        if self.cirriculum :
            self.train_target_signatures = [-3, -1, 1, 3]
        else :
            self.train_target_signatures = [
                sig for sig in range(-self.max_braid_length+10, self.max_braid_length-9) if sig % 2 == 1
            ]
        self.test_target_signatures = [
            sig for sig in range(-self.max_braid_length+10, self.max_braid_length-9) if (sig % 2 == 0) and sig != 0
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
        # initialize the braid word with a random generator, could also experiment with starting with it empty
        self.braid_word = [Integer(np.random.choice([i for i in range(-self.braid_index+1, self.braid_index)  if i != 0]))]
        self.braid_word_lk_rep = self.generator_lk_matrices[self.braid_word[0]]
        self.link = Link(self.B(self.braid_word))
        self.current_signature = self.link.signature()

        # self.target_signature = target_signature
        self.target_signature = np.random.choice(self.target_signatures)

        # for reward_type = 'dense', setting this equal to target signature gives a large negative reward
        # on the first step and smaller rewards on subsequent steps. It is unsed for reward_type = 'sparse'
        self.t_minus_1_signature = self.target_signature 

        # calculate and return the state + info dict
        if self.state_rep == 'Lawrence-Krammer' :
            return self.braid_word_lk_rep, {}
        elif self.state_rep == 'LK_plus_signatures':
            return np.concatenate([np.array([self.target_signature, self.current_signature]),
                                   self.braid_word_lk_rep.flatten()]), {}

    def step(self, action):
        # if a generator was appended, i.e. if the STOP token wasn't selected
        if action < (self.braid_index-1)*2 :
            # I followed the convention used in https://arxiv.org/abs/1610.05744 for ordering the generators
            # sigma_1, sigma_2, ..., sigma_n, sigma_{-1}, sigma_{-2}, ..., sigma_{-n}
            generator = (action % (self.braid_index-1)) + 1
            if action >= self.braid_index - 1 :
                generator = -generator
            self.braid_word.append(Integer(generator))
            self.braid_word_lk_rep = self.braid_word_lk_rep @ self.generator_lk_matrices[generator]
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

        else : # the STOP action was selected
            terminated = True
            truncated = False

            # calculate the reward
            if self.reward_type == 'dense' :
                reward = 0
            elif self.reward_type == 'sparse' :
                reward = -np.abs(self.current_signature - self.target_signature)

        if self.state_rep == 'Lawrence-Krammer' :
            state = self.braid_word_lk_rep
        elif self.state_rep == 'LK_plus_signatures' :
            state = np.concatenate([np.array([self.target_signature, self.current_signature]),
                                    self.braid_word_lk_rep.flatten()])

        return state, reward, terminated, truncated, {}

    def render(self):
        self.link.plot()

    def close(self):
        pass

    
