import os
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from environment import JobRoutingGymEnv, Config
from PPO import PPOAgent   
import torch
np.set_printoptions(suppress=True, precision=2,
                    formatter={'float_kind': lambda x: f"{x:.2f}"})

N_EPISODES = 3000

def evaluate(agent: PPOAgent, env: gym.Env, n_episodes=10):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        while not done:
            action_vec, _ = agent.select_action(obs)
            action = int(action_vec[0])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ret += reward
        returns.append(ret)
    return float(np.mean(returns)), float(np.std(returns))


def get_new_logdir(base_dir="runs/tb", prefix="PPO_JobRouting"):
    os.makedirs(base_dir, exist_ok=True)
    i = 1
    while True:
        logdir = os.path.join(base_dir, f"{prefix}{i}")
        if not os.path.exists(logdir):
            return logdir
        i += 1


def main():
    logdir = get_new_logdir()
    writer = SummaryWriter(logdir)
    print(f"[TensorBoard] logging to {logdir}")

    cfg = Config()
    cfg.SEED = 0
    env = JobRoutingGymEnv(cfg)
    eval_env = JobRoutingGymEnv(cfg)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    NUM_JOBS = cfg.NUM_JOBS
    total_timesteps = NUM_JOBS * N_EPISODES

    agent = PPOAgent(state_dim=state_dim, action_dims=[n_actions])

    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    ep_count = 0
    ep_loss_sum, ep_loss_count = 0.0, 0   

    for t in range(1, total_timesteps + 1):
        action_vec, log_prob = agent.select_action(obs)
        action = int(action_vec[0])

        obs2, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store_transition((obs, action_vec, reward, obs2, done, log_prob.item()))

        obs = obs2
        ep_ret += reward
        ep_len += 1

        if done:
            ep_count += 1
            avg_reward = ep_ret / ep_len

            learn_time = agent.update()
            if learn_time is not None:
                ep_loss_sum += learn_time
                ep_loss_count += 1

            avg_loss = ep_loss_sum / ep_loss_count if ep_loss_count > 0 else 0.0

            writer.add_scalar("train/avg_reward", avg_reward, ep_count)
            writer.add_scalar("train/loss", avg_loss, ep_count)

            print(f"[Episode {ep_count}] reward={avg_reward:.2f}, loss={avg_loss:.4f}")

            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0
            ep_loss_sum, ep_loss_count = 0.0, 0  

    mean_ret, std_ret = evaluate(agent, eval_env, n_episodes=20)
    print(f"[EVAL] Return mean={int(mean_ret):,} ± {int(std_ret):,}")

    torch.save(agent.policy.state_dict(), "runs/ppo_agent.pt")
    writer.close()


if __name__ == "__main__":
    main()
