import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)
target_actor = Actor(state_dim, action_dim)
target_critic = Critic(state_dim, action_dim)

# Initialize target networks weights with original networks
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=0.0001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

replay_buffer = ReplayBuffer(1000000)
gamma = 0.99
tau = 0.001
batch_size = 64

# List to store rewards for each episode
episode_rewards = []

# Training loop
for episode in range(400):
    state = env.reset()
    episode_reward = 0
    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = actor(state_tensor).detach().numpy() + np.random.normal(0, 0.1, size=(action_dim,))
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        # Training from replay buffer
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            state_batch = torch.tensor(batch[0], dtype=torch.float32)
            action_batch = torch.tensor(batch[1], dtype=torch.float32).unsqueeze(1).squeeze(-1)
            reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1)
            next_state_batch = torch.tensor(batch[3], dtype=torch.float32)
            done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1)

            with torch.no_grad():
                next_actions = target_actor(next_state_batch)
                target_q_values = target_critic(next_state_batch, next_actions)
                q_targets = reward_batch + gamma * (1 - done_batch) * target_q_values

            q_values = critic(state_batch, action_batch)
            critic_loss = F.mse_loss(q_values, q_targets)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -critic(state_batch, actor(state_batch)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update target networks
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        state = next_state
        episode_reward += reward
        if done:
            episode_rewards.append(episode_reward)
            print(f"Episode {episode}: Total reward = {episode_reward}")
            break

# Visualization
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward vs Episode for DDPG on Pendulum-v1')
plt.show()
