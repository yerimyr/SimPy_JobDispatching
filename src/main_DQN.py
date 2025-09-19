import os
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from environment import JobRoutingGymEnv, Config
from DQN import DQNAgent, BATCH_SIZE

N_EPISODES = 1000 

def evaluate(agent: DQNAgent, env: gym.Env, n_episodes=10):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ret += reward
        returns.append(ret)
    return float(np.mean(returns)), float(np.std(returns))


def get_new_logdir(base_dir="runs/tb", prefix="DQN_JobRouting"):
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

    cfg = Config(SEED=0)
    env = JobRoutingGymEnv(cfg)
    eval_env = JobRoutingGymEnv(Config(SEED=123))

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    NUM_JOBS = cfg.NUM_JOBS

    total_timesteps = NUM_JOBS * N_EPISODES

    agent = DQNAgent(state_dim=state_dim, action_dim=n_actions)

    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    ep_count = 0
    ep_loss_sum, ep_loss_count = 0.0, 0   # loss 기록용

    for t in range(1, total_timesteps + 1):
        action = agent.select_action(obs)
        obs2, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.push(obs, action, reward, obs2, done)

        loss = agent.update(batch_size=BATCH_SIZE)
        if loss is not None:
            ep_loss_sum += loss
            ep_loss_count += 1

        obs = obs2
        ep_ret += reward
        ep_len += 1

        if done:
            ep_count += 1
            avg_reward = ep_ret / ep_len 
            avg_loss = ep_loss_sum / ep_loss_count if ep_loss_count > 0 else 0.0

            writer.add_scalar("train/avg_reward", avg_reward, ep_count)
            writer.add_scalar("train/loss", avg_loss, ep_count)

            print(f"[Episode {ep_count}] reward={avg_reward:.2f}")

            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0
            ep_loss_sum, ep_loss_count = 0.0, 0   # 초기화

    mean_ret, std_ret = evaluate(agent, eval_env, n_episodes=20)
    print(f"[EVAL] Return mean={int(mean_ret):,} ± {int(std_ret):,}")

    agent.save("runs/dqn_agent.pt")
    writer.close()

if __name__ == "__main__":
    main()
