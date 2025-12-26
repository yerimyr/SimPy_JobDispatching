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


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
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
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = GAMMA

        self.device = torch.device(device)

        # networks
        self.q_network = QNet(state_dim, action_dim).to(self.device)
        self.target_network = QNet(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # exploration
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
            slope = (
                self.exploration_final_eps - self.exploration_initial_eps
            ) / self.exploration_timesteps
            self.exploration_rate = (
                self.exploration_initial_eps + slope * self.global_step
            )

    def select_action(self, state, greedy=False):
        self.update_exploration()

        if (not greedy) and random.random() < self.exploration_rate:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(torch.argmax(q_values, dim=1).item())


    def update(self, batch_size=BATCH_SIZE):
        if len(self.replay_buffer) < batch_size:
            return None

        s, a, r, ns, d = self.replay_buffer.sample(batch_size)

        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.q_network(s).gather(1, a)

        with torch.no_grad():
            q_next = self.target_network(ns).max(dim=1, keepdim=True)[0]
            q_target = r + (1 - d) * self.gamma * q_next

        loss = nn.SmoothL1Loss()(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        self.loss = loss.item()
        self.global_step += 1

        if self.global_step % TARGET_UPDATE_INTERVAL == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return self.loss


    def save(self, path):
        torch.save(
            {
                "policy": self.q_network.state_dict(),
                "target": self.target_network.state_dict(),
                "opt": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q_network.load_state_dict(ckpt["policy"])
        self.target_network.load_state_dict(
            ckpt.get("target", ckpt["policy"])
        )
        self.optimizer.load_state_dict(ckpt["opt"])
        self.global_step = ckpt.get("global_step", 0)
        self.target_network.eval()
