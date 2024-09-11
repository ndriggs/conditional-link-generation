import gymnasium as gym
from gymnasium import spaces
import snappy
import numpy as np
import torch

class LinkBuilderEnv(gym.Env):
    def __init__(self, braid_index: int = 7, state: str = 'Lawrence-Krammer', reward_type: str = 'dense', 
                 curiousity: bool = False):
        super(LinkBuilderEnv, self).__init__()

        if braid_index < 3 : 
            raise ValueError(f"Invalid param: {braid_index}. 'braid_index' parameter must be greater than 2")

        if state not in ['Lawrence-Krammer', 'invariants'] :
            raise ValueError(f"Invalid param: {state}. 'state' parameter must be one of 'Lawrence-Krammer' or 'invariants'.")

        if reward_type not in ['dense', 'sparse'] :
            raise ValueError(f"Invalid param: {reward_type}. 'reward_type' parameter must be one of 'dense' or 'sparse'")

        self.braid_index = braid_index
        self.B = BraidGroup(self.braid_index)
        self.max_braid_length = 75 # somewhat arbitrary, still computes signature for longer braids

        # randomly pick a target signature between -target signature min and max for each episode 
        # ideally there should be a "teacher" creating a "curriculum," carefully  
        # selecting which tasks to train on 
        self.target_signature_min = -np.round(self.max_braid_length/2.5)
        self.target_signature_max = np.round(self.max_braid_length/2.5)
        self.target_signature = np.random.randint(self.target_signature_min, self.target_signature_max+1)

        # braid_index = 3 would give 5 actions: {sigma_1, sigma_2, sigma_{-1}, sigma_{-2}, STOP}
        self.action_space = spaces.Discrete((self.braid_index-1)*2 + 1) 

        # create the observation space
        if state == 'Lawrence-Krammer' :
            self.lk_matrix_size = self.braid_index*(self.braid_index-1)//2 # code credit: Mark Hughes

            # the LK rep can blow up for large braid words 
            observation_space = gym.spaces.Box(low=-2e13, high=6e13, 
                                               shape=(self.lk_matrix_size, self.lk_matrix_size), 
                                               dtype=np.float64)
        
        # initialize the braid word with a random generator, could also experiment with starting with it empty
        self.braid_word = [np.random.choice([i for i in range(-self.braid_index+1, self.braid_index)  if i != 0])]
        self.link = Link(self.B(self.braid_word))
        self.current_signature = self.link.signature()

        # compute the Lawrence-Krammer representation for each braid group generator
        # code courtesy of Mark Hughes
        self.generator_lk_matrices = {}
        for sigma_i in range(-self.braid_index+1,self.braid_index):
            if np.sign(sigma_i) == -1:
                self.generator_lk_matrices[sigma_i]=np.linalg.inv(self.lk_rep(self.braid_index,np.abs(sigma_i)))
            elif np.sign(sigma_i) == 1:
                self.generator_lk_matrices[sigma_i]=self.lk_rep(self.braid_index,np.abs(sigma_i))

        self.braid_word_lk_rep = self.generator_lk_matrices[self.braid_word[0]]

    def reset(self, target_signature: int):
        # initialize the braid word with a random generator, could also experiment with starting with it empty
        self.braid_word = [np.random.choice([i for i in range(-self.braid_index+1, self.braid_index)  if i != 0])]
        self.braid_word_lk_rep = self.generator_lk_matrices[self.braid_word[0]]
        self.link = Link(self.B(self.braid_word))
        self.current_signature = self.link.signature()
        
        self.target_signature = target_signature

        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        # if a generator was appended, i.e. if the STOP token wasn't selected
        if action < (self.braid_index-1)*2 :
            # I followed the convention used in https://arxiv.org/abs/1610.05744 for ordering the generators
            # sigma_1, sigma_2, ..., sigma_n, sigma_{-1}, sigma_{-2}, ..., sigma_{-n}
            generator = (action % (self.braid_index-1)) + 1
            if action >= self.braid_index - 1 :
                generator = -generator
            self.braid_word.append(generator)
            self.braid_word_lk_rep = self.braid_word_lk_rep @ self.generator_lk_matrices[generator]

            terminated = False 
            if len(self.braid_word) >= self.max_braid_length :
                truncated = True
            else :
                truncated = False

            # calculate the reward
            if reward_type == 'dense' :
                ### max reward 0 or 100???? ###
            elif reward_type == 'sparse' :
                reward = 0

        else : # the STOP action was selected
            terminated = True
            truncated = False

            # calculate the reward
            if reward_type == 'dense' :
                reward = 0
            elif reward_type == 'sparse' :
                reward = -np.abs(self.current_signature - self.target_signature)

        return np.array([self.state], dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        self.link.plot()

    def close(self):
        pass

    # computes the Lawrence-Krammer representation of sigma_k (k > 0) with a braid index of n
    # lk_rep and index functions courtesy of Mark Hughes
    def lk_rep(self,n,k):
        M=np.zeros((n*(n-1)//2,n*(n-1)//2))
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

    def get_signatures() :
        return np.array([self.current_signature, self.target_signature])

    def get_braid_word() :
        return self.braid_word
