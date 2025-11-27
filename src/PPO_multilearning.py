import time
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from environment import Config

# ====== 하이퍼파라미터 로딩 ======
cfg = Config()
STATE_DIM = cfg.STATE_DIM
N_ACTIONS = cfg.N_ACTIONS
PPO_LEARNING_RATE = cfg.LEARNING_RATE
PPO_GAMMA = cfg.GAMMA
PPO_CLIP_EPSILON = cfg.CLIP_EPSILON
PPO_UPDATE_STEPS = cfg.UPDATE_STEPS
PPO_GAE_LAMBDA = cfg.GAE_LAMBDA
PPO_ENT_COEF = cfg.ENT_COEF
PPO_VF_COEF = cfg.VF_COEF
PPO_MAX_GRAD_NORM = cfg.MAX_GRAD_NORM
PPO_BATCH_SIZE = cfg.BATCH_SIZE
PPO_HIDDEN = cfg.HIDDEN_SIZE


def _orthogonal_init(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dims: List[int], hidden_size: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.action_heads = nn.ModuleList([nn.Linear(hidden_size, d) for d in action_dims])

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.actor.apply(lambda m: _orthogonal_init(m, gain=nn.init.calculate_gain('tanh')))
        for h in self.action_heads:
            _orthogonal_init(h, gain=0.01)
        self.critic.apply(lambda m: _orthogonal_init(m, gain=nn.init.calculate_gain('tanh')))

    def forward(self, x: torch.Tensor):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        feat = self.actor(x)
        probs = [torch.softmax(h(feat), dim=-1) for h in self.action_heads]  
        v = self.critic(x).squeeze(-1)

        if single:
            probs = [p[0] for p in probs]
            v = v[0]
        return probs, v


class PPOAgent:
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dims: List[int] = None,
        lr: float = PPO_LEARNING_RATE,
        gamma: float = PPO_GAMMA,
        clip_epsilon: float = PPO_CLIP_EPSILON,
        update_steps: int = PPO_UPDATE_STEPS,
        gae_lambda: float = PPO_GAE_LAMBDA,
        ent_coef: float = PPO_ENT_COEF,
        vf_coef: float = PPO_VF_COEF,
        max_grad_norm: float = PPO_MAX_GRAD_NORM,
        hidden_size: int = PPO_HIDDEN,
        batch_size: int = PPO_BATCH_SIZE,
        device: str = "cpu",
        target_kl: float = 0.02,
        value_clip_eps: float = 0.2,
        ent_decay: float = 0.995,
        lr_decay: Optional[float] = 0.999
    ):
        if action_dims is None:
            action_dims = [N_ACTIONS]

        self.device = torch.device(device)

        self.gamma = float(gamma)
        self.clip_epsilon = float(clip_epsilon)
        self.update_steps = int(update_steps)
        self.gae_lambda = float(gae_lambda)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.batch_size = int(batch_size)

        self.target_kl = float(target_kl)
        self.value_clip_eps = float(value_clip_eps)
        self.ent_decay = float(ent_decay)
        self.lr_decay = float(lr_decay) if lr_decay is not None else None

        self.policy = ActorCritic(state_dim, action_dims, hidden_size=hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = (
            optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
            if self.lr_decay is not None else None
        )

        self.memory: List[Tuple] = []
        self.learn_time = 0.0

    def select_action(self, state):
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs, _ = self.policy(st)
        acts, logps = [], []
        for p in probs:
            cat = Categorical(p)
            a = cat.sample()
            acts.append(int(a.item()))
            logps.append(cat.log_prob(a))
        return np.array(acts, dtype=np.int64), torch.sum(torch.stack(logps))

    def store_transition(self, tr):
        self.memory.append(tr)

    @staticmethod
    def _compute_gae(r, v, v_next, d, gamma, lam):
        T = r.shape[0]
        adv = torch.zeros(T, device=r.device)
        gae = 0.0
        for t in range(T - 1, -1, -1):
            nd = 1.0 - d[t]  # not done
            delta = r[t] + gamma * v_next[t] * nd - v[t]
            gae = delta + gamma * lam * nd * gae
            adv[t] = gae
        return adv

    def update(self):
        if not self.memory:
            return 0.0

        s, a, r, s2, d, lp = zip(*self.memory)
        S = torch.as_tensor(np.array(s), dtype=torch.float32, device=self.device)
        A = torch.as_tensor(np.array(a), dtype=torch.long, device=self.device)
        R = torch.as_tensor(np.array(r), dtype=torch.float32, device=self.device)
        S2 = torch.as_tensor(np.array(s2), dtype=torch.float32, device=self.device)
        D = torch.as_tensor(np.array(d), dtype=torch.float32, device=self.device)
        LP = torch.as_tensor(np.array(lp), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, V = self.policy(S)
            _, V2 = self.policy(S2)
            V2 = V2 * (1.0 - D)

        ADV = self._compute_gae(R, V, V2, D, self.gamma, self.gae_lambda)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        V_TGT = R + self.gamma * V2

        t0 = time.time()
        idx = np.arange(S.size(0))
        for _ in range(self.update_steps):
            np.random.shuffle(idx)
            approx_kl_epoch = 0.0
            count_kl = 0

            for i in range(0, len(idx), self.batch_size):
                b = idx[i:i + self.batch_size]
                st, act, oldp = S[b], A[b], LP[b].detach()
                advb, vtgt = ADV[b], V_TGT[b].detach()
                vold = V[b].detach()

                probs, vpred = self.policy(st)

                cat = Categorical(probs[0])
                logp_new = cat.log_prob(act[:, 0])
                ratio = torch.exp(logp_new - oldp)

                surr1 = ratio * advb
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advb
                policy_loss = -torch.min(surr1, surr2).mean()

                v_unclipped = (vpred - vtgt).pow(2)
                v_clipped = (vold + (vpred - vold).clamp(-self.value_clip_eps, self.value_clip_eps) - vtgt).pow(2)
                value_loss = torch.max(v_unclipped, v_clipped).mean()

                entropy = cat.entropy().mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl = (oldp - logp_new).mean()
                    approx_kl_epoch += float(kl.item())
                    count_kl += 1

            if count_kl > 0:
                mean_kl = approx_kl_epoch / count_kl
                if mean_kl > self.target_kl:
                    break

        self.learn_time = time.time() - t0

        self.clip_epsilon = max(0.10, self.clip_epsilon * 0.995)
        self.ent_coef = max(0.0, self.ent_coef * self.ent_decay)
        if self.scheduler is not None:
            self.scheduler.step()

        self.memory.clear()
        return self.learn_time

    def compute_loss(self) -> torch.Tensor:
        if not self.memory:
            raise ValueError("Memory is empty.")

        s, a, r, s2, d, lp = zip(*self.memory)
        S = torch.as_tensor(np.array(s), dtype=torch.float32, device=self.device)
        A = torch.as_tensor(np.array(a), dtype=torch.long, device=self.device)
        R = torch.as_tensor(np.array(r), dtype=torch.float32, device=self.device)
        S2 = torch.as_tensor(np.array(s2), dtype=torch.float32, device=self.device)
        D = torch.as_tensor(np.array(d), dtype=torch.float32, device=self.device)
        LP = torch.as_tensor(np.array(lp), dtype=torch.float32, device=self.device)

        _, V = self.policy(S)
        _, V2 = self.policy(S2)
        V2 = V2 * (1.0 - D)

        ADV = self._compute_gae(R, V, V2, D, self.gamma, self.gae_lambda)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        V_TGT = R + self.gamma * V2

        probs, Vp = self.policy(S)
        cat = Categorical(probs[0])  
        logp_new = cat.log_prob(A[:, 0])

        ratio = torch.exp(logp_new - LP)
        surr1 = ratio * ADV
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * ADV
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = (Vp - V_TGT).pow(2).mean()
        entropy = cat.entropy().mean()

        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        return loss

    def compute_gradients(self) -> Dict[str, torch.Tensor]:
        self.optimizer.zero_grad()
        loss = self.compute_loss()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        grads = {n: (p.grad.detach().clone()) for n, p in self.policy.named_parameters() if p.requires_grad}
        return grads

    def apply_gradients(self, avg_gradients: Dict[str, torch.Tensor]):
        self.optimizer.zero_grad()
        for name, param in self.policy.named_parameters():
            if param.requires_grad and name in avg_gradients:
                param.grad = avg_gradients[name].to(self.device).detach().clone()
        self.optimizer.step()

        self.clip_epsilon = max(0.10, self.clip_epsilon * 0.995)
        self.ent_coef = max(0.0, self.ent_coef * self.ent_decay)
        if self.scheduler is not None:
            self.scheduler.step()

    def to(self, device: str):
        self.device = torch.device(device)
        self.policy.to(self.device)
