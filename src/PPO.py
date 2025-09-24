import time, math
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from environment import Config

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
    """
    Initialize the given layer using orthogonal initialization.

    Args:
        m (nn.Module): A PyTorch module, typically an nn.Linear layer, 
            whose weights and biases will be initialized.
        gain (float, optional): Scaling factor applied to the orthogonal 
            matrix. Default is 1.0.
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ActorCritic(nn.Module):
    """
    Actor-Critic neural network architecture for reinforcement learning.

    This model contains two main components:
    - Actor: Outputs action probabilities for each action dimension.
    - Critic: Outputs the state-value estimate.

    Args:
        state_dim (int): Dimension of the input state space.
        action_dims (List[int]): List specifying the number of discrete 
            actions for each action dimension (for MultiDiscrete action space).
        hidden_size (int, optional): Number of hidden units per layer. 
            Default is 64.
    """
    def __init__(self, state_dim: int, action_dims: List[int], hidden_size: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.action_heads = nn.ModuleList([nn.Linear(hidden_size, d) for d in action_dims])
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.actor.apply(lambda m: _orthogonal_init(m, gain=nn.init.calculate_gain('tanh')))
        for h in self.action_heads: _orthogonal_init(h, gain=0.01)
        self.critic.apply(lambda m: _orthogonal_init(m, gain=nn.init.calculate_gain('tanh')))

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the Actor-Critic model.

        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, state_dim) 
                or (state_dim,) for a single state.

        Returns:
            probs (List[torch.Tensor]): A list of action probability distributions, 
                one per action dimension.
            v (torch.Tensor): Estimated state-value(s), scalar per input state.
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        feat = self.actor(x)
        probs = [torch.softmax(h(feat), dim=-1) for h in self.action_heads]  # len==1
        v = self.critic(x).squeeze(-1)
        if single:
            probs = [p[0] for p in probs]
            v = v[0]
        return probs, v

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent implementation.

    This agent trains an Actor-Critic neural network using PPO, a policy gradient 
    algorithm that stabilizes learning with clipping and advantage estimation.

    Args:
        state_dim (int): Dimension of the input state space.
        action_dims (List[int]): List specifying the number of discrete actions per action dimension.
        lr (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        clip_epsilon (float): Clipping parameter for PPO objective.
        update_steps (int): Number of epochs per update.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation (GAE).
        ent_coef (float): Coefficient for entropy regularization (encourages exploration).
        vf_coef (float): Coefficient for value function loss.
        max_grad_norm (float): Maximum gradient norm for clipping.
        hidden_size (int): Hidden layer size for Actor-Critic network.
        batch_size (int): Mini-batch size for updates.
        device (str): Device for computation ('cpu' or 'cuda').
        target_kl (float): Target KL-divergence threshold for early stopping.
        value_clip_eps (float): Clipping range for value function updates.
        ent_decay (float): Decay factor for entropy coefficient.
        lr_decay (float, optional): Learning rate decay factor.
    """
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
        """
        Select an action based on the current policy.

        Args:
            state (np.ndarray): Current environment state.

        Returns:
            acts (np.ndarray): Sampled action(s) from the policy distribution.
            logps (torch.Tensor): Log probability of the chosen action(s).
        """
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
        """
        Store a transition in memory.

        Args:
            tr (Tuple): A tuple (state, action, reward, next_state, done, log_prob).
        """
        self.memory.append(tr)

    @staticmethod
    def _compute_gae(r, v, v_next, d, gamma, lam):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            r (torch.Tensor): Rewards.
            v (torch.Tensor): State value estimates.
            v_next (torch.Tensor): Next state value estimates.
            d (torch.Tensor): Done flags (1 if terminal state, else 0).
            gamma (float): Discount factor.
            lam (float): GAE lambda parameter.

        Returns:
            adv (torch.Tensor): Advantage estimates.
        """
        T = r.shape[0]
        adv = torch.zeros(T, device=r.device)
        gae = 0.0
        for t in range(T - 1, -1, -1):
            nd = 1.0 - d[t] 
            delta = r[t] + gamma * v_next[t] * nd - v[t]
            gae = delta + gamma * lam * nd * gae
            adv[t] = gae
        return adv

    def update(self):
        """
        Perform a PPO update using stored transitions.

        Returns:
            learn_time (float): Time taken for the update (seconds).

        Process:
            1. Convert memory into tensors.
            2. Compute advantages using GAE.
            3. Optimize policy and value function with clipping.
            4. Apply entropy regularization for exploration.
            5. Use KL early stopping to prevent instability.
            6. Optionally decay entropy coefficient and learning rate.
        """
        if not self.memory:
            return 0.0

        s, a, r, s2, d, lp = zip(*self.memory)
        S  = torch.as_tensor(np.array(s),  dtype=torch.float32, device=self.device)
        A  = torch.as_tensor(np.array(a),  dtype=torch.long,    device=self.device)   
        R  = torch.as_tensor(np.array(r),  dtype=torch.float32, device=self.device)
        S2 = torch.as_tensor(np.array(s2), dtype=torch.float32, device=self.device)
        D  = torch.as_tensor(np.array(d),  dtype=torch.float32, device=self.device)
        LP = torch.as_tensor(np.array(lp), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, V  = self.policy(S)    
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
                v_clipped   = (vold + (vpred - vold).clamp(-self.value_clip_eps, self.value_clip_eps) - vtgt).pow(2)
                value_loss  = torch.max(v_unclipped, v_clipped).mean()

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

    def to(self, device: str):
        """
        Move the policy network and tensors to a specified device.

        Args:
            device (str): 'cpu' or 'cuda'.
        """
        self.device = torch.device(device)
        self.policy.to(self.device)
