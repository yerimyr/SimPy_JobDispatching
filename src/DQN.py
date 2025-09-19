# dqn.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 5000
TARGET_UPDATE_INTERVAL = 500
MAX_GRAD_NORM = 10

EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05
EXPLORATION_FRACTION = 0.1  
TOTAL_TIMESTEPS = 20_000    

#DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  
        states, actions, rewards, next_states, dones = zip(*batch)  

        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(actions).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(dones).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.update_step = 0

        self.q_network = QNet(state_dim, action_dim).to(DEVICE)
        self.target_network = QNet(state_dim, action_dim).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())  
        self.target_network.eval()  

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        self.exploration_initial_eps = EXPLORATION_INITIAL_EPS
        self.exploration_final_eps = EXPLORATION_FINAL_EPS
        self.exploration_timesteps = int(TOTAL_TIMESTEPS * EXPLORATION_FRACTION)
        self.global_step = 0
        self.exploration_rate = self.exploration_initial_eps

        self.loss = None

    def update_exploration(self):
        if self.global_step >= self.exploration_timesteps:
            self.exploration_rate = self.exploration_final_eps
        else:
            slope = (self.exploration_final_eps - self.exploration_initial_eps) / self.exploration_timesteps
            self.exploration_rate = self.exploration_initial_eps + slope * self.global_step

    def select_action(self, state, greedy=False):
        self.update_exploration()
        if (not greedy) and (random.random() < self.exploration_rate):
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch_size=BATCH_SIZE):
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        self.loss = loss.item()
        self.global_step += 1

        if self.global_step % TARGET_UPDATE_INTERVAL == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return self.loss

    def save(self, path: str):
        torch.save({
            "policy": self.q_network.state_dict(),
            "target": self.target_network.state_dict(),
            "opt": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or DEVICE)
        self.q_network.load_state_dict(ckpt["policy"])
        self.target_network.load_state_dict(ckpt.get("target", ckpt["policy"]))
        self.optimizer.load_state_dict(ckpt["opt"])
        self.global_step = ckpt.get("global_step", 0)
        self.target_network.eval()
