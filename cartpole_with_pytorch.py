import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import numpy as np

env = gym.make("CartPole-v1")

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# nn.Module is the base class for all neural network modules in PyTorch. 
# It provides a convenient way to define and manage the parameters of a neural network, as well as to implement the forward pass of the network.
# By subclassing nn.Module, you can create your own custom neural network architectures and easily integrate them into your training loop.
class DQN(nn.Module):
    def __init__(self):
        # super() is a built-in function in Python that allows you to call methods from a parent class. 
        # In the context of a class that inherits from nn.Module, super().__init__() initializes the parent class.
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, n_actions)
        )

    def forward(self, x):
        return self.net(x)

model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

gamma = 0.99
epsilon = 1.0
episodes = 500
eval_episodes = 10

model.train()

for episode in range(episodes):
    state, _ = env.reset()
    # print("Initial state:", state)
    state = torch.tensor(state, dtype=torch.float32)
    # print("Tensor state:", state)

    done = False
    truncated = False
    while not done and not truncated:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = torch.argmax(model(state)).item()

        next_state, reward, done, truncated, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # compute target
        with torch.no_grad():
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            if done or truncated:
                target = reward_tensor
            else:
                target = reward_tensor + gamma * torch.max(model(next_state))

        output = model(state)[action]

        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    epsilon *= 0.995

print("Training complete")

model.eval()
eval_rewards = []

for eval_episode in range(eval_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    truncated = False
    total_reward = 0

    while not done and not truncated:
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()

        next_state, reward, done, truncated, _ = env.step(action)
        state = torch.tensor(next_state, dtype=torch.float32)
        total_reward += reward

    eval_rewards.append(total_reward)
    print(f"Evaluation episode {eval_episode + 1}: reward = {total_reward}")

average_reward = sum(eval_rewards) / len(eval_rewards)
print(f"Average evaluation reward over {eval_episodes} episodes: {average_reward:.2f}")