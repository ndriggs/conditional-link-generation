import gymnasium as gym
from gymnasium import spaces
import snappy

class LinkBuilderEnv(gym.Env):
    def __init__(self, braid_index=6):
        super(LinkBuilderEnv, self).__init__()

        # braid_index = 3 would give 5 actions: {sigma_1, sigma_2, sigma_{-1}, sigma_{-2}, STOP}
        self.action_space = spaces.Discrete((braid_index-1)*2 + 1) 
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([100]), dtype=np.float32)
        
        # initialize the braid word with a random generator, could also experiment with starting with it empty
        self.braid_word = [np.random.choice([i for i in range(-braid_index+1, 3)  if i != 0])]

    def reset(self):
        self.state = np.random.randint(0, 100)  # Reset environment state randomly
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
