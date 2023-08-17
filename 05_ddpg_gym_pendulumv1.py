import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
import numpy as np
import matplotlib.pyplot as plt

# Environment setup
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)
target_actor = Actor(state_dim, action_dim)
target_critic = Critic(state_dim, action_dim)
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

episode_rewards = []

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(10000)

# Hyperparameters
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)
gamma = 0.99
tau = 0.005  # Soft update parameter
noise_std = 0.1

# Training loop
for episode in range(400):
    state = env.reset()
    episode_reward = 0
    while True:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = actor(state_tensor).numpy() + np.random.normal(0, noise_std, size=(1,))

        next_state, reward, done, _ = env.step(action)

        replay_buffer.push(state, action, reward, next_state, done)

        if len(replay_buffer) > 64:
            transitions = replay_buffer.sample(64)
            batch = Transition(*transitions)
            state_batch = torch.tensor(batch.state, dtype=torch.float32)
            
            action_batch = torch.tensor(batch.action, dtype=torch.float32).unsqueeze(1).squeeze(-1)

            reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
            next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)
            done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).float()


            with torch.no_grad():
                next_actions = target_actor(next_state_batch)
                target_q_values = target_critic(next_state_batch, next_actions)

                q_targets = reward_batch + gamma * (1 - done_batch) * target_q_values

            q_values = critic(state_batch, action_batch)
            critic_loss = F.mse_loss(q_values, q_targets)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_actions = actor(state_batch)
            actor_loss = -critic(state_batch, actor_actions).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

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






