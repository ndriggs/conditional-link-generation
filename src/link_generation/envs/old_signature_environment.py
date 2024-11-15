import gymnasium as gym
from gymnasium import spaces
from typing import Any
from sage.all import BraidGroup, Link, Integer
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
import snappy
import numpy as np

class LinkBuilderEnv(gym.Env):

    metadata = {"render_modes": ["knot_diagram", "braid_word"], "render_fps": 2}

    def __init__(self, reward_type: str, braid_index: int = 7, state_rep: str = 'LK_plus_signatures', 
                 curiousity: bool = False, render_mode='knot_diagram'):
        super(LinkBuilderEnv, self).__init__()

        if braid_index < 3 : 
            raise ValueError(f"Invalid param: {braid_index}. 'braid_index' parameter must be greater than 2")

        if state_rep not in ['Lawrence-Krammer', 'invariants', 'LK_plus_signatures'] :
            raise ValueError(f"Invalid param: {state_rep}. 'state_rep' parameter must be one of 'Lawrence-Krammer', 'invariants', or 'LK_plus_signatures'.")

        if reward_type not in ['dense', 'sparse'] :
            raise ValueError(f"Invalid param: {reward_type}. 'reward_type' parameter must be one of 'dense' or 'sparse'")

        self.braid_index = braid_index
        self.B = BraidGroup(self.braid_index)
        self.max_braid_length = 75 # somewhat arbitrary, still computes signature for longer braids
        self.lk_matrix_size = self.braid_index*(self.braid_index-1)//2 # code credit: Mark Hughes
        self.reward_type = reward_type
        self.num_envs = 1 # so StableBaselines3 can use VecEnv
        self.render_mode = render_mode

        # randomly pick a target signature between -target signature min and max for each episode 
        # ideally there should be a "teacher" creating a "curriculum," carefully  
        # selecting which tasks to train on 
        # self.target_signature_min = -np.round(self.max_braid_length/2.1)
        # self.target_signature_max = np.round(self.max_braid_length/2.1)
        # self.target_signature = np.random.randint(self.target_signature_min, self.target_signature_max+1)
        self.target_signatures = [-11,-10,-9,9,10,11]

        # braid_index = 3 would give 5 actions: {sigma_1, sigma_2, sigma_{-1}, sigma_{-2}, STOP}
        self.action_space = spaces.Discrete((self.braid_index-1)*2 + 1) 

        # create the observation space
        self.state_rep = state_rep
        if self.state_rep == 'Lawrence-Krammer' :
            # the LK rep can blow up for large braid words 
            self.observation_space = spaces.Box(low=-2e13, high=6e13, 
                                                shape=(self.lk_matrix_size, self.lk_matrix_size), 
                                                dtype=np.float64)

        elif self.state_rep == 'LK_plus_signatures' :
            # Bounds for the first 2 dimensions representing target signature and current signature
            low_signature = np.array([min(self.target_signatures), -self.max_braid_length], dtype=np.float64)
            high_signature = np.array([max(self.target_signatures), self.max_braid_length], dtype=np.float64)

            # Bounds for the remaining LK rep dimensions 
            low_lk_rep = np.full(self.lk_matrix_size**2, -2e13, dtype=np.float64)
            high_lk_rep = np.full(self.lk_matrix_size**2, 6e13, dtype=np.float64)

            # Combine the low and high bounds for all dimensions
            low = np.concatenate([low_signature, low_lk_rep])
            high = np.concatenate([high_signature, high_lk_rep])

            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # compute the Lawrence-Krammer representation for each braid group generator
        # code courtesy of Mark Hughes
        self.generator_lk_matrices = {}
        for sigma_i in range(-self.braid_index+1,self.braid_index):
            if np.sign(sigma_i) == -1:
                self.generator_lk_matrices[sigma_i]=np.linalg.inv(self.lk_rep(self.braid_index,np.abs(sigma_i)))
            elif np.sign(sigma_i) == 1:
                self.generator_lk_matrices[sigma_i]=self.lk_rep(self.braid_index,np.abs(sigma_i))

        if curiousity :
            pass

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

    # computes the Lawrence-Krammer representation of sigma_k (k > 0) with a braid index of n
    # lk_rep and index functions courtesy of Mark Hughes
    def lk_rep(self,n,k):
        M=np.zeros((n*(n-1)//2,n*(n-1)//2), dtype=np.float64)  
        q=np.sqrt(2)
        t=np.pi
        for i in range(1,n):
            for j in range(i+1,n+1):
                if (k<i-1)or(j<k):
                    M[self.index(n,i,j),self.index(n,i,j)]=1
                elif k==i-1:
                    M[self.index(n,i-1,j),self.index(n,i,j)]= 1
                    M[self.index(n,i,j),self.index(n,i,j)] = 1-q
                elif (k==i) and (k<j-1):
                    M[self.index(n,i,i+1),self.index(n,i,j)] = t*q*(q - 1)
                    M[self.index(n,i+1,j),self.index(n,i,j)] = q
                elif (k==i) and (k ==j-1):
                    M[self.index(n,i,j),self.index(n,i,j)] = t*q*q
                elif (i<k) and (k<j - 1):
                    M[self.index(n,i,j),self.index(n,i,j)] = 1 
                    M[self.index(n,k,k+1),self.index(n,i,j)] = t*q**(k - i)*(q - 1)**2
                elif (k==j-1):
                    M[self.index(n,i,j-1),self.index(n,i,j)] = 1
                    M[self.index(n,j-1,j),self.index(n,i,j)] = t*q**(j-i)*(q - 1)
                elif (k==j):
                    M[self.index(n,i,j),self.index(n,i,j)]=1-q
                    M[self.index(n,i,j+1),self.index(n,i,j)]=q
        return M

    # used in the lk_rep function
    def index(self,n,i,j):
        return int((i-1)*(n-i/2)+j-i-1)

    def get_signatures(self) :
        return np.array([self.current_signature, self.target_signature])

    def get_braid_word(self) :
        return self.braid_word
