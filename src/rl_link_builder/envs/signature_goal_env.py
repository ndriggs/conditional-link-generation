import gymnasium as gym
from gymnasium import spaces
import gymnasium_robotics
from typing import Any
from sage.all import BraidGroup, Link, Integer
import snappy
import numpy as np
import warnings
warnings.simplefilter("error")


class LinkBuilderEnv(gymnasium_robotics.core.GoalEnv):

    metadata = {"render_modes": ["knot_diagram", "braid_word", "rgb_array"], "render_fps": 2}

    def __init__(self, braid_index: int = 7, state_rep: str = 'LK_plus_signatures', 
                 curiousity: bool = False, render_mode: str = 'knot_diagram'):
        super(LinkBuilderEnv, self).__init__()

        if braid_index < 3 : 
            raise ValueError(f"Invalid param: {braid_index}. 'braid_index' parameter must be greater than 2")

        if state_rep not in ['Lawrence-Krammer', 'invariants', 'LK_plus_signatures'] :
            raise ValueError(f"Invalid param: {state_rep}. 'state_rep' parameter must be one of 'Lawrence-Krammer', 'invariants', or 'LK_plus_signatures'.")

        # if reward_type not in ['dense', 'sparse'] :
        #     raise ValueError(f"Invalid param: {reward_type}. 'reward_type' parameter must be one of 'dense' or 'sparse'")

        self.braid_index = braid_index
        self.B = BraidGroup(self.braid_index)
        self.max_braid_length = 45 # 75 somewhat arbitrary, still computes signature for longer braids
        self.lk_matrix_size = self.braid_index*(self.braid_index-1)//2 # code credit: Mark Hughes
        self.num_envs = 1 # so StableBaselines3 can use VecEnv
        self.render_mode = render_mode

        # randomly pick a target signature between -target signature min and max for each episode 
        # ideally there should be a "teacher" creating a "curriculum," carefully  
        # selecting which tasks to train on 
        # self.target_signature_min = -np.round(self.max_braid_length/2.1)
        # self.target_signature_max = np.round(self.max_braid_length/2.1)
        # self.target_signature = np.random.randint(self.target_signature_min, self.target_signature_max+1)
        # self.target_signatures = [-11,-10,-9,9,10,11]
        self.target_signatures = [sig for sig in np.arange(-18,19) if sig != 0]

        # braid_index = 3 would give 4 actions: {sigma_1, sigma_2, sigma_{-1}, sigma_{-2}}
        self.action_space = spaces.Discrete((self.braid_index-1)*2) 

        # create the observation space
        self.state_rep = state_rep
        if self.state_rep == 'Lawrence-Krammer' :
            # the LK rep can blow up for large braid words 
            obs_space = spaces.Box(low=-2e13, high=6e13, 
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

            obs_space = spaces.Box(low=low, high=high, dtype=np.float64)

        goal_space = spaces.Box(low=-self.max_braid_length, high=self.max_braid_length, shape=(1,), dtype=np.int32)

        self.observation_space = spaces.Dict({
            'observation': obs_space,
            'achieved_goal': goal_space,
            'desired_goal': goal_space
        })

        # compute the Lawrence-Krammer representation for each braid group generator
        # code courtesy of Mark Hughes
        self.generator_lk_matrices = {}
        for sigma_i in range(-self.braid_index+1,self.braid_index):
            if np.sign(sigma_i) == -1:
                self.generator_lk_matrices[sigma_i]=np.linalg.inv(self.lk_rep(self.braid_index,np.abs(sigma_i)))
            elif np.sign(sigma_i) == 1:
                self.generator_lk_matrices[sigma_i]=self.lk_rep(self.braid_index,np.abs(sigma_i))

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None): # target_signature: int
        # initialize the braid word with a random generator, could also experiment with starting with it empty
        self.braid_word = [Integer(np.random.choice([i for i in range(-self.braid_index+1, self.braid_index)  if i != 0]))]
        self.braid_word_lk_rep = self.generator_lk_matrices[self.braid_word[0]]
        self.link = Link(self.B(self.braid_word))
        self.current_signature = self.link.signature()
        self.achieved_goal = np.array([self.current_signature], dtype=np.int32)
        # self.t_minus_1_signature = self.current_signature 
        # self.target_signature = target_signature
        self.target_signature = np.random.choice(self.target_signatures)
        self.desired_goal = np.array([self.target_signature], dtype=np.int32)

        # calculate the state
        if self.state_rep == 'Lawrence-Krammer' :
            state = self.braid_word_lk_rep
        elif self.state_rep == 'LK_plus_signatures':
            state =  np.concatenate([np.array([self.target_signature, self.current_signature]),
                                     self.braid_word_lk_rep.flatten()])
        
        observation = {
            'observation': state,
            'achieved_goal': self.achieved_goal,
            'desired_goal': self.desired_goal
        }

        return observation, {}


    def step(self, action):
        # if a generator was appended, i.e. if the STOP token wasn't selected
        # if action < (self.braid_index-1)*2 :
        # I followed the convention used in https://arxiv.org/abs/1610.05744 for ordering the generators
        # sigma_1, sigma_2, ..., sigma_n, sigma_{-1}, sigma_{-2}, ..., sigma_{-n}
        generator = (action % (self.braid_index-1)) + 1
        if action >= (self.braid_index - 1) :
            generator = -generator
        self.braid_word.append(Integer(generator))
        try: 
            self.braid_word_lk_rep = self.braid_word_lk_rep @ self.generator_lk_matrices[generator]
            terminated_info = {'RuntimeWarning': False}
        except RuntimeWarning as e: 
            print('RuntimeWarning message: ', e)
            print('braid word length: ', len(self.braid_word))
            print('max: ', np.max(self.braid_word_lk_rep), 'min: ', np.min(self.braid_word_lk_rep))
            terminated_info = {'RuntimeWarning': True}
        self.link = Link(self.B(self.braid_word))
        self.current_signature = self.link.signature()
        self.achieved_goal = np.array([self.current_signature], dtype=np.int32)
            
            # terminated_info = {'stop_action_selected': False}

        # else : # the STOP action was selected
            # terminated_info = {'stop_action_selected': True}
            
        truncated_info = {'braid_word_length': len(self.braid_word)}
        reward = self.compute_reward(self.achieved_goal, self.desired_goal, {})
        terminated = self.compute_terminated(self.achieved_goal, self.desired_goal, terminated_info)
        truncated = self.compute_truncated(self.achieved_goal, self.desired_goal, truncated_info)

        if self.state_rep == 'Lawrence-Krammer' :
            state = self.braid_word_lk_rep
        elif self.state_rep == 'LK_plus_signatures' :
            state = np.concatenate([np.array([self.target_signature, self.current_signature]),
                                    self.braid_word_lk_rep.flatten()])

        obs = {
            'observation': state,
            'achieved_goal': self.achieved_goal,
            'desired_goal': self.desired_goal
        }

        info = {
            'RuntimeWarning': terminated_info['RuntimeWarning'],
            'braid_length': len(self.braid_word),
            'signature': self.current_signature
        }
        if terminated :
            info['time_taken'] = np.abs(self.target_signature) / (len(self.braid_word) - 1)
        elif truncated :
            info['missed_target'] = self.target_signature

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info) :
        return (achieved_goal == desired_goal).all(axis=-1).astype(np.float32)

    def compute_terminated(self, achieved_goal, desired_goal, info) :
        if all(achieved_goal == desired_goal) :
            return True
        # elif info['stop_action_selected'] :
        #     return True
        elif info['RuntimeWarning'] :
            return True
        else :
            return False

    def compute_truncated(self, achieved_goal, desired_goal, info):
        if info['braid_word_length'] >= self.max_braid_length :
            return True
        else :
            return False

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

    def get_signatures() :
        return np.array([self.current_signature, self.target_signature])

    def get_braid_word() :
        return self.braid_word
