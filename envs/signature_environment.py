import gymnasium as gym
from gymnasium import spaces
import snappy
import numpy as np
import torch

class LinkBuilderEnv(gym.Env):
    def __init__(self, braid_index: int = 7, state : str = 'Lawrence-Krammer', curiousity : bool = False):
        super(LinkBuilderEnv, self).__init__()

        if braid_index < 3 : 
            raise ValueError(f"Invalid param: {braid_index}. 'braid_index' parameter must be greater than 2")

        if state not in ['Lawrence-Krammer', 'braid word', 'invariants'] :
            raise ValueError(f"Invalid param: {state}. 'state' parameter must be one of 'Lawrence-Krammer', 'braid word', or 'invariants'.")

        self.braid_index = braid_index
        self.B = BraidGroup(self.braid_index)
        self.max_braid_length = 75 # somewhat arbitrary, still computes signature for longer braids

        # randomly pick a target signature between -22 and 22 for each episode 
        # ideally there should be a "teacher" creating a "curriculum," carefully  
        # selecting which tasks to train on 
        self.target_signature_min = -np.round(self.max_braid_length/2.5)
        self.target_signature_max = np.round(self.max_braid_length/2.5)
        self.target_signature = np.random.randint(self.target_signature_min, self.target_signature_max+1)

        # braid_index = 3 would give 5 actions: {sigma_1, sigma_2, sigma_{-1}, sigma_{-2}, STOP}
        self.action_space = spaces.Discrete((self.braid_index-1)*2 + 1) 

        # create the observation space
        # signature will stay pretty small, but the LK rep can blow up for large braid words
        self.lk_matrix_size = self.braid_index*(self.braid_index-1)//2 # code credit: Mark Hughes
        
        # Bounds for the first 2 dimensions representing target signature and current signature
        low_signature = np.array([self.target_signature_min, -self.max_braid_length], dtype=np.float64)
        high_signature = np.array([self.target_signature_max, self.max_braid_length], dtype=np.float64)

        # Bounds for the remaining LK rep dimensions 
        low_lk_rep = np.full(self.lk_matrix_size**2, -2e12, dtype=np.float64)
        high_lk_rep = np.full(self.lk_matrix_size**2, 6e12, dtype=np.float64)

        # Combine the low and high bounds for all dimensions
        low = np.concatenate([low_signature, low_lk_rep])
        high = np.concatenate([high_signature, high_lk_rep])
        
        observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
        
        # initialize the braid word with a random generator, could also experiment with starting with it empty
        self.braid_word = [np.random.choice([i for i in range(-self.braid_index+1, self.braid_index)  if i != 0])]
        self.link = Link(self.B(self.braid_word))

        # compute the Lawrence-Krammer representation for each braid group generator
        # code courtesy of Mark Hughes
        self.generator_lk_matrices = {}
        for sigma_i in range(-self.braid_index+1,self.braid_index):
            if np.sign(sigma_i) == -1:
                self.generator_lk_matrices[sigma_i]=np.linalg.inv(self.lk_rep(self.braid_index,np.abs(sigma_i)))
            elif np.sign(sigma_i) == 1:
                self.generator_lk_matrices[sigma_i]=self.lk_rep(self.braid_index,np.abs(sigma_i))

        self.braid_word_lk_rep = self.generator_lk_matrices[self.braid_word[0]]
        self.state = self.get_state()

    def reset(self):
        # initialize the braid word with a random generator, could also experiment with starting with it empty
        self.braid_word = [np.random.choice([i for i in range(-self.braid_index+1, self.braid_index)  if i != 0])]
        self.braid_word_lk_rep = self.generator_lk_matrices[self.braid_word[0]]
        self.link = Link(self.B(self.braid_word))
        self.state = self.get_state()

        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        # Define how action affects state and how the reward is computed
        if action == 0:
            self.state -= 1
        else:
            self.state += 1

        # Compute reward and whether the environment is done
        reward = -abs(self.state - 50)  # Reward is based on proximity to the value 50
        done = self.state < 0 or self.state > 100  # Done if state is out of bounds
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
    
    def get_state() :

