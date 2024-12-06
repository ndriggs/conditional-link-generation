import gymnasium as gym
from gymnasium import spaces
from typing import Any
from sage.all import BraidGroup, Link, Integer
# from link_generation.models.curiousity_models import GNN
import numpy as np

class SigDetEnv(gym.Env):

    def __init__(self, state_rep:str, reward_type:str, seed:int, max_braid_length: int, braid_index:int = 7, w1:float = 0.7):
        '''
        params:
        reward_type: either 'dense' which will give the agent a reward everytime the signature
                     changes or 'sparse' which will only give the agent a reward when it finishes
                     the episode
        seed: the random seed to be used for numpy, affects the initial starting generator
        braid_index: the number of strands to braid
        w1: weights how important signature is vs. log determinant in the reward
        '''
        super(SigDetEnv, self).__init__()

        if braid_index < 3 : 
            raise ValueError(f"Invalid param: {braid_index}. 'braid_index' parameter must be greater than 2")

        if reward_type not in ['dense', 'sparse'] :
            raise ValueError(f"Invalid param: {reward_type}. 'reward_type' parameter must be one of 'dense' or 'sparse'")

        np.random.seed(seed)
        self.braid_index = braid_index 
        self.B = BraidGroup(self.braid_index)
        self.max_braid_length = max_braid_length # somewhat arbitrary, computes signature for braids as long as 80
        self.state_rep = state_rep
        self.reward_type = reward_type
        self.num_envs = 1 # so StableBaselines3 can use VecEnv  
        self.w1 = w1

        # braid_index = 3 would give 5 actions: {sigma_1, sigma_2, sigma_1^{-1}, sigma_2^{-1}, STOP}
        # 0 represents the STOP action, 1 through braid_index -1 are the generators, braid_index through 
        # 2*braid_index -2 are the inverses
        self.action_space = spaces.Discrete((self.braid_index-1)*2 + 1, start=-self.braid_index+1) 

        # create the observation space
        if state_rep == 'braid' :
            self.observation_space = spaces.Box(low=-self.braid_index+1, high=self.braid_index-1, 
                                                shape=(self.max_braid_length,), dtype=np.int32)
        elif state_rep == 'ohe' :
            self.observation_space = spaces.MultiBinary(2*(self.braid_index-1)*self.max_braid_length)
        else : 
            raise ValueError(f'Only "braid" and "ohe" state representations are currently supported, not {state_rep}')
        # can also try including current sig and det


        # if self.curiousity :
        #     self.model = GNN(hidden_channels=32, num_heads=8, num_layers=4, dropout=0,
        #                      classification=False, both=True, ohe_inverses=True, 
        #                      double_features=True)
            

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None): # target_signature: int
        if seed is not None :
            np.random.seed(seed)
        
        # initialize the braid word with a random generator or inverse
        self.braid_word = [Integer(np.random.choice([i for i in range(-self.braid_index+1, self.braid_index)  if i != 0]))]
        self.link = Link(self.B(self.braid_word))
        self.current_signature = self.link.signature()
        self.current_det = self.link.determinant()
        self.t_minus_1_signature = self.current_signature
        self.t_minus_1_det = self.current_det

        if self.state_rep == 'braid' :
            state = self.braid_word_to_braid_state()
        elif self.state_rep == 'ohe' :
            state = self.braid_word_to_ohe_state()

        return state, {}


    def step(self, action):
        # the STOP token was selected
        if action == 0 :
            terminated = True
            truncated = False

            # calculate the reward
            if self.reward_type == 'dense' :
                reward = 0
            elif self.reward_type == 'sparse' :
                if self.link.is_knot() :
                    reward = self.w1*np.abs(self.current_signature) - (1-self.w1)*np.log1p(self.current_det)
                else : 
                    reward = -1

            info = {
                'signature': self.current_signature,
                'determinant': self.current_det,
                'braid_word': self.braid_word,
                'is_knot': int(self.link.is_knot())
            }
        
        else : 
            if self.action_space.contains(action) :
                self.braid_word.append(Integer(action))
            else : # is a generator inverse
                self.braid_word.append(Integer(-(action - (self.braid_index-1))))
            self.link = Link(self.B(self.braid_word))
            self.current_signature = self.link.signature()
            self.current_det = self.link.determinant()

            terminated = False 
            if len(self.braid_word) >= self.max_braid_length :
                truncated = True
                info = {
                    'signature': self.current_signature,
                    'determinant': self.current_det,
                    'braid_word': self.braid_word,
                    'is_knot': int(self.link.is_knot())
                }
            else :
                truncated = False
                info = {}

            # calculate the reward
            if self.reward_type == 'dense' :
                reward = self.w1*(np.abs(self.current_signature) - np.abs(self.t_minus_1_signature)) \
                    + (1-self.w1)*(np.log1p(self.t_minus_1_det) - np.log1p(self.current_det))
            elif self.reward_type == 'sparse' :
                if truncated :
                    if self.link.is_knot() :
                        reward = self.w1*np.abs(self.current_signature) - (1-self.w1)*np.log1p(self.current_det)
                    else : 
                        reward = -1
                else : 
                    reward = int(self.link.is_knot())

            self.t_minus_1_signature = self.current_signature
            self.t_minus_1_det = self.current_det

        if self.state_rep == 'braid' :
            state = self.braid_word_to_braid_state()
        elif self.state_rep == 'ohe' :
            state = self.braid_word_to_ohe_state()

        return state, reward, terminated, truncated, info

    def render(self):
        self.link.plot()

    def close(self):
        pass

    def braid_word_to_braid_state(self) :
        state = np.zeros(self.max_braid_length, dtype=np.int32)
        state[:len(self.braid_word)] = self.braid_word
        return state

    def braid_word_to_ohe_state(self) :
        state = np.zeros(2*(self.braid_index-1)*self.max_braid_length)
        idxs = (np.abs(self.braid_word) + \
                ((1-np.sign(self.braid_word))*((self.braid_index-1)/2)).astype(int) - 1) + \
                    np.arange(0,2*(self.braid_index-1)*len(self.braid_word), 2*(self.braid_index-1))
        state[idxs] = np.ones(len(self.braid_word))
        return state
    
