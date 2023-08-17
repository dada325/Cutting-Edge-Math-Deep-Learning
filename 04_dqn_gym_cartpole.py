import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt

# Environment setup
env = gym.make('CartPole-v1')

# Q-Network definition
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
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

# Hyperparameters
lr = 0.01
gamma = 0.99
epsilon = 1.0
num_episodes = 1000
batch_size = 100
target_update_freq = 100
buffer_size = 10000

# DQN setup
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
q_network = QNetwork(input_dim, output_dim)
target_network = QNetwork(input_dim, output_dim)
target_network.load_state_dict(q_network.state_dict())
optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
loss_fn = nn.MSELoss()
replay_buffer = ReplayBuffer(buffer_size)

episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = q_network(state_tensor).argmax().item()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        replay_buffer.push(state, action, reward, next_state, done)

        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            with torch.no_grad():
                target_q_values = target_network(next_states)
                max_target_q_values, _ = target_q_values.max(dim=1)
                targets = rewards + (1 - dones) * gamma * max_target_q_values

            predicted_q_values = q_network(states).gather(1, actions.unsqueeze(-1))
            loss = loss_fn(predicted_q_values, targets.unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state

    episode_rewards.append(total_reward)

    if episode % target_update_freq == 0:
        target_network.load_state_dict(q_network.state_dict())
    
    epsilon *= 0.995
    epsilon = max(0.01, epsilon)

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

# Evaluate
def evaluate_policy(policy, episodes=10):
    total_rewards = 0.0
    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = policy(state_tensor).argmax().item()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards += episode_reward
    average_reward = total_rewards / episodes
    return average_reward

avg_reward = evaluate_policy(q_network)
print(f"Average reward over evaluation episodes: {avg_reward}")

# Save the model
torch.save(q_network.state_dict(), "dqn_model.pth")

# Visualization
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward vs Episode')
plt.show()
