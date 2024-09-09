import gymnasium as gym
from gymnasium import spaces
import snappy
import numpy as np

class LinkBuilderEnv(gym.Env):
    def __init__(self, braid_index=6):
        super(LinkBuilderEnv, self).__init__()

        self.braid_index = braid_index

        # braid_index = 3 would give 5 actions: {sigma_1, sigma_2, sigma_{-1}, sigma_{-2}, STOP}
        self.action_space = spaces.Discrete((braid_index-1)*2 + 1) 
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([100]), dtype=np.float32)
        
        # initialize the braid word with a random generator, could also experiment with starting with it empty
        self.braid_word = [np.random.choice([i for i in range(-braid_index+1, braid_index)  if i != 0])]

        # compute the Lawrence-Krammer representation for each braid group generator
        self.lk_matrix_size = self.braid_index*(self.braid_index-1)//2
        self.generator_LK_matrices = {}
        for sigma_i in range(-self.braid_index+1,self.braid_index):
            sgn=np.sign(sigma_i)
            if sgn==-1:
                self.generator_LK_matrices[jjj]=np.linalg.inv(self.LKrep(self.braid_index,np.abs(sigma_i)))
            elif sgn==1:
                self.generator_LK_matrices[jjj]=self.LKrep(self.braid_index,np.abs(sigma_i))

    def reset(self):
        # initialize the braid word with a random generator, could also experiment with starting with it empty
        self.braid_word = [np.random.choice([i for i in range(-braid_index+1, braid_index)  if i != 0])]

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
        print(f"State: {self.state}")

    def close(self):
        pass
