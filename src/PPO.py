import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from environment import *
import time

class ActorCritic(nn.Module):
    """
    Actor-Critic model for PPO with MultiDiscrete action space.

    Args:
        state_dim: Dimension of the state space.
        action_dims: List containing the number of discrete actions per action dimension.
        hidden_size: Number of neurons in hidden layers.
    """
    def __init__(self, state_dim, action_dims, hidden_size=Config.HIDDEN_SIZE):
        super(ActorCritic, self).__init__()
        self.action_dims = action_dims

        # Policy Network (Actor)
        self.actor_fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.action_heads = nn.ModuleList([nn.Linear(hidden_size, dim) for dim in action_dims])  # MultiDiscrete

        # Value Network (Critic)
        self.critic_fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state):
        """
        Forward pass through the Actor-Critic network.

        Args:
            state: Current state of the environment.

        Returns:
            action_probs: Probability distributions for MultiDiscrete action dimensions.
            value: Estimated state value.
        """
        actor_features = self.actor_fc(state)
        action_probs = [torch.softmax(head(actor_features), dim=-1) for head in self.action_heads]  # MultiDiscrete
        value = self.critic_fc(state)
        return action_probs, value

class PPOAgent:
    """
    PPO Agent with MultiDiscrete action space handling.

    This class implements the Proximal Policy Optimization (PPO) algorithm 
    for environments with MultiDiscrete action spaces. The agent consists 
    of an Actor-Critic model and uses the Generalized Advantage Estimation (GAE)
    method for efficient policy updates.

    Args:
        state_dim: Dimension of the state space.
        action_dims: Number of discrete actions for each action dimension.
        lr: Learning rate for the optimizer.
        gamma: Discount factor for future rewards.
        clip_epsilon: Clipping range for PPO.
        update_steps: Number of training epochs per update.
    """
    def __init__(self, state_dim, action_dims, lr=Config.LEARNING_RATE, gamma=Config.GAMMA, clip_epsilon=Config.CLIP_EPSILON, update_steps=Config.UPDATE_STEPS,
                gae_lambda=Config.GAE_LAMBDA, ent_coef=Config.ENT_COEF, vf_coef=Config.VF_COEF, max_grad_norm=Config.MAX_GRAD_NORM, hidden_size=Config.HIDDEN_SIZE):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_steps = update_steps
        self.gae_lambda = gae_lambda          
        self.ent_coef = ent_coef              
        self.vf_coef = vf_coef                
        self.max_grad_norm = max_grad_norm    
        self.device = Config.DEVICE
    
        self.policy = ActorCritic(state_dim, action_dims, hidden_size=hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []
        
    def select_action(self, state):
        """
        Selects an action for MultiDiscrete environments.

        Args:
            state: Current state of the environment.

        Returns:
            actions: Selected actions for each action dimension.
            log_prob: Summed log probability of the selected actions because of multidiscrete environment.
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_probs, _ = self.policy(state)
        
        actions = []
        log_probs = []
        for dist in action_probs:
            categorical_dist = Categorical(dist)
            action = categorical_dist.sample()
            actions.append(action.item())
            log_probs.append(categorical_dist.log_prob(action))
        
        return np.array(actions), torch.sum(torch.stack(log_probs)) 
    
    def store_transition(self, transition):
        """
        Stores a transition in memory.
        
        Args:
        transition: A tuple containing:
            - state: The current state.
            - action: The action taken.
            - reward: The reward received after taking the action.
            - next_state: The state after taking the action.
            - done: Whether the episode has ended.
            - log_prob: The log probability of the selected action.
        """
        self.memory.append(transition)
    
    def update(self):
        """
        Performs PPO update using stored experience.

        This function processes stored transitions, computes advantages,
        and updates the policy and value networks using PPO loss.
        """
        if not self.memory:
            print("Memory is empty.")
            return
        
        states, actions, rewards, next_states, dones, log_probs_old = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        log_probs_old = torch.tensor(np.array(log_probs_old), dtype=torch.float32, device=self.device)

        _, values = self.policy(states)
        _, next_values = self.policy(next_states)
        not_dones = (1 - dones).unsqueeze(1)
        next_values = (next_values * not_dones).clone()

        advantages = self._compute_gae(rewards, values.detach().squeeze(), self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        value_target = rewards + self.gamma * next_values.view(-1).detach()

        batch_size = Config.BATCH_SIZE  
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        start_time = time.time()
        for _ in range(self.update_steps):
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices].detach().clone()
                batch_value_target = value_target[batch_indices].detach().clone()

                action_probs, values_new = self.policy(batch_states)
                
                log_probs_new = []
                for j, dist in enumerate(action_probs):
                    categorical_dist = Categorical(dist)
                    log_probs_new.append(categorical_dist.log_prob(batch_actions[:, j]))
                log_probs_new = torch.sum(torch.stack(log_probs_new), dim=0)
                
                ratio = torch.exp(log_probs_new - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values_new.view(-1), batch_value_target)
                
                entropy = torch.stack([
                    Categorical(dist).entropy().mean() for dist in action_probs
                ]).mean()
                entropy_loss = -entropy
                
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)  
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                self.learn_time=time.time()-start_time
        
        self.clip_epsilon = max(0.1, self.clip_epsilon * 0.995)
        self.memory.clear()
        
        return self.learn_time
    
    def _compute_gae(self, rewards, values, gamma, lambda_):
        """
        Computes Generalized Advantage Estimation (GAE) for PPO.

        Args:
            rewards: Rewards obtained from environment.
            values: Estimated values of the states.
            gamma: Discount factor.
            lambda_: Smoothing factor for GAE.

        Returns:
            torch.Tensor: Computed advantage estimates.
        """
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0
        N = len(rewards)
        for t in reversed(range(N)):
            v      = values[t]
            next_v = values[t+1] if t < N-1 else torch.tensor(0.0, device=self.device)
            delta  = rewards[t] + gamma * next_v - v
            gae    = delta + gamma * lambda_ * gae
            advantages[t] = gae
        return advantages
    
    # Parallel Learning functions
    def compute_loss(self):
        if not self.memory:
            raise ValueError("Memory is empty.")

        states, actions, rewards, next_states, dones, log_probs_old = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        log_probs_old = torch.tensor(np.array(log_probs_old), dtype=torch.float32, device=self.device)

        _, values = self.policy(states)
        _, next_values = self.policy(next_states)
        not_dones = (1 - dones).unsqueeze(1)
        next_values = (next_values * not_dones).clone()

        advantages = self._compute_gae(rewards, values.detach().squeeze(), self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        value_target = rewards + self.gamma * next_values.view(-1).detach()
        
        batch_size = Config.BATCH_SIZE  
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        
        for _ in range(self.update_steps):
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices].detach().clone()
                batch_value_target = value_target[batch_indices].detach().clone()
                action_probs, values_new = self.policy(batch_states)
                log_probs_new = []
                for j, dist in enumerate(action_probs):
                    categorical_dist = Categorical(dist)
                    log_probs_new.append(categorical_dist.log_prob(batch_actions[:, j]))
                log_probs_new = torch.sum(torch.stack(log_probs_new), dim=0)

                ratio = torch.exp(log_probs_new - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values_new.view(-1), batch_value_target)

                entropy = torch.stack([
                    Categorical(dist).entropy().mean() for dist in action_probs
                ]).mean()
                entropy_loss = -entropy

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
        return loss
    
    # Parallel Learning functions
    def apply_gradients(self, avg_gradients):
        self.optimizer.zero_grad()
        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                param.grad = avg_gradients[name].to(param.device).clone()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.clip_epsilon = max(0.1, self.clip_epsilon * 0.999)  # 천천히 줄이거나 아예 고정


    # Parallel Learning functions
    def compute_gradients(self):
        self.optimizer.zero_grad()
        loss = self.compute_loss()  
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        gradients = {name: param.grad.clone() for name, param in self.policy.named_parameters() if param.requires_grad}
        return gradients