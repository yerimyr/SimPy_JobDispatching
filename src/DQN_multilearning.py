# DQN_multilearning.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        """transition 저장"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """랜덤 샘플링"""
        batch = random.sample(self.buffer, batch_size)

        s, a, r, ns, d = map(np.array, zip(*batch))
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(ns, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )


# ==========================================
# Q-Network
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


# ==========================================
# DQN Agent (멀티러닝 대응)
# ==========================================
class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden=64,
        lr=1e-3,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
        batch_size=10,
        min_buffer_size=1000,
        target_update_interval=1000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_decay = eps_decay

        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_interval = target_update_interval
        self.learn_step = 0

        # 네트워크
        self.q_network = QNetwork(state_dim, action_dim, hidden)
        self.target_network = QNetwork(state_dim, action_dim, hidden)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # replay buffer
        self.replay_buffer = ReplayBuffer(capacity=50000)

    # --------------------------------------
    # 행동 선택 (epsilon-greedy)
    # --------------------------------------
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        s = torch.tensor(state, dtype=torch.float32)
        q_values = self.q_network(s)
        return int(q_values.argmax().item())

    # --------------------------------------
    # epsilon decay
    # --------------------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    # --------------------------------------
    # compute gradients (워커에서 호출)
    # --------------------------------------
    def compute_gradients(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.replay_buffer) < self.min_buffer_size:
            raise ValueError("Replay buffer too small")

        # minibatch sample
        s, a, r, ns, d = self.replay_buffer.sample(batch_size)

        # Q(s, a)
        q_values = self.q_network(s)
        q_a = q_values.gather(1, a.view(-1, 1)).squeeze(1)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.target_network(ns).max(dim=1)[0]
            q_target = r + self.gamma * q_next * (1 - d)

        loss = nn.MSELoss()(q_a, q_target)

        # compute gradients only
        self.optimizer.zero_grad()
        loss.backward()

        # clip
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)

        # gradient dict 추출
        grads = {}
        for name, param in self.q_network.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().clone()

        # target network 업데이트 step 증가
        self.learn_step += 1
        if self.learn_step % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return grads

    # --------------------------------------
    # apply gradients (서버에서 호출)
    # --------------------------------------
    def apply_gradients(self, grad_dict):
        self.optimizer.zero_grad()

        # apply gradients
        for name, param in self.q_network.named_parameters():
            if param.requires_grad and name in grad_dict:
                param.grad = grad_dict[name].clone()

        self.optimizer.step()
        self.target_network.load_state_dict(self.q_network.state_dict())

    # --------------------------------------
    # utils
    # --------------------------------------
    def to(self, device):
        self.q_network.to(device)
        self.target_network.to(device)
