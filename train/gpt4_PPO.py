from gymnasium.envs.registration import register

register(
    id='LinkBuilderEnv-v0',  
    entry_point='envs.signature_environment:LinkBuilderEnv',  
)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import gymnasium as gym

device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network for Actor and Critic
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.hidden_dim = 100
        self.shared_base = nn.Sequential(nn.Linear(num_inputs, self.hidden_dim), nn.ReLU())
        self.policy_layer = nn.Linear(self.hidden_dim, num_actions)
        self.value_layer = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, x):
        base = self.shared_base(x)
        policy = self.policy_layer(base)
        value = self.value_layer(base)
        return policy, value

# Compute GAE
def compute_gae(next_value, rewards, masks, values, gamma=1, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

# Function for taking an action using the policy
def act_and_evaluate(model, state, memory):
    state = torch.FloatTensor(state).to(device)
    policy, value = model(state)
    policy_dist = torch.distributions.Categorical(logits=policy)
    action = policy_dist.sample()
    memory.actions.append(action)
    memory.states.append(state)
    memory.logprobs.append(policy_dist.log_prob(action))
    memory.values.append(value)
    return action.cpu().numpy()

# The PPO Algorithm
def ppo_update(model, optimizer, scheduler, memory, ppo_epochs=10, mini_batch_size=15): # mini_batch_size=5
    for epoch in range(ppo_epochs):
        sampled_indices = np.random.permutation(len(memory.rewards))

        for i in range(len(memory.rewards) // mini_batch_size):
            indices = sampled_indices[i*mini_batch_size:(i+1)*mini_batch_size]
            states = torch.stack([memory.states[i] for i in indices]).squeeze(-1)
            actions = torch.stack([memory.actions[i] for i in indices])
            old_logprobs = torch.stack([memory.logprobs[i] for i in indices])

            policy, value = model(states)
            policy_dist = torch.distributions.Categorical(logits=policy)
            logprobs = policy_dist.log_prob(actions)
            entropy = policy_dist.entropy().mean()
            
            # PPO ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Compute advantage estimates
            rewards = [memory.rewards[i] for i in indices]
            next_value = model(torch.FloatTensor(memory.next_states).to(device))[1]
            returns = compute_gae(next_value, rewards, memory.masks, memory.values)
            returns = torch.tensor(returns).to(device)
            advantage = returns - value.detach()
            
            # PPO objective
            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - value).pow(2).mean()

            optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss - 0.01 * entropy).backward()
            optimizer.step()
            
            # Update learning rate
            scheduler.step()

# A simple memory to store trajectories
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.masks = []  # Masks for terminals
        self.next_states = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.masks[:]
        del self.next_states[:]

# Noam learning rate scheduler
def noam_scheduler(optimizer, warmup_steps, factor, model_dim):
    step_count = 0
    def _lr_scheduler(optimizer):
        nonlocal step_count
        step_count += 1
        return factor * (model_dim ** -0.5) * min(step_count ** -0.5, step_count * warmup_steps**-1.5)
    
    return LambdaLR(optimizer, _lr_scheduler)

# Main loop for training
def train(env_name, num_epochs=10, steps_per_epoch=100): # num_epochs=1000 steps_per_epoch=2048
    env = gym.make(env_name)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    model = ActorCritic(num_inputs, num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = noam_scheduler(optimizer, warmup_steps=100, factor=1.0, model_dim=model.hidden_dim) # warmup_steps=4000

    memory = Memory()
    rewards_per_episode = []

    for epoch in range(num_epochs):
        state, info = env.reset()
        rewards_this_episode = []
        for step in range(steps_per_epoch):
            action = act_and_evaluate(model, state, memory)
            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards_this_episode.append(reward)
            done = terminated or truncated
            memory.rewards.append(reward)
            memory.masks.append(1 - float(done))
            memory.next_states.append(next_state)
            state = next_state
            if done:
                state, info = env.reset()
                rewards_per_episode.append(sum(rewards_this_episode))
                rewards_this_episode = []

        ppo_update(model, optimizer, scheduler, memory)
        memory.clear()

    plt.figure(figsize=(10, 6))
    plt.plot(rewards_per_episode)
    plt.ylabel('Reward per Episode')
    plt.xlabel('Episode Number')
    plt.savefig('cumulative_reward.png')

    torch.save(model.state_dict(), 'model.pt')

    

if __name__ == "__main__":
    env_name = "LinkBuilderEnv-v0"
    train(env_name)