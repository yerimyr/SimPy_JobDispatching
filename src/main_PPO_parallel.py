"""
import os
import time
import multiprocessing
import torch
from torch.utils.tensorboard import SummaryWriter
from PPO import PPOAgent
from environment import JobRoutingGymEnv, Config

# 학습 파라미터
N_MULTIPROCESS = 5
N_EPISODES = 1000
NUM_JOBS = Config.NUM_JOBS

def get_next_exp_id(base_dir="./runs/tb", prefix="parallel_ppo_grad_"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        int(name.replace(prefix, "")) 
        for name in os.listdir(base_dir) 
        if name.startswith(prefix) and name.replace(prefix, "").isdigit()
    ]
    return max(existing) + 1 if existing else 1

EXP_ID = get_next_exp_id()
LOG_DIR = f"./runs/tb/parallel_ppo_grad_{EXP_ID}"
main_writer = SummaryWriter(log_dir=LOG_DIR)

def build_model(env):
    obs, _ = env.reset()
    state_dim = len(obs)
    action_dims = [env.action_space.n]
    model = PPOAgent(
        state_dim=state_dim,
        action_dims=action_dims,
        lr=Config.LEARNING_RATE,
        gamma=Config.GAMMA,
        clip_epsilon=Config.CLIP_EPSILON,
        update_steps=Config.UPDATE_STEPS
    )
    return model

def simulation_worker(core_index, model_state_dict, episode_idx):
    env = JobRoutingGymEnv(Config())
    agent = build_model(env)
    agent.policy.load_state_dict(model_state_dict)

    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0

    start_sampling = time.time()
    for _ in range(NUM_JOBS):
        action_vec, log_prob = agent.select_action(obs)
        action = int(action_vec[0])
        obs2, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store_transition((obs, action_vec, reward, obs2, done, log_prob.item()))
        obs = obs2
        ep_ret += reward
        ep_len += 1
        if done:
            break
    sampling_time = time.time() - start_sampling

    # gradient 계산
    start_update = time.time()
    gradients = agent.compute_gradients()
    learn_time = time.time() - start_update

    avg_reward = ep_ret / ep_len if ep_len > 0 else 0.0

    return core_index, sampling_time, learn_time, avg_reward, gradients, ep_len, episode_idx

def worker_wrapper(args):
    return simulation_worker(*args)

def average_gradients(gradient_dicts):
    avg_grad = {}
    for key in gradient_dicts[0].keys():
        avg_grad[key] = sum(d[key] for d in gradient_dicts) / len(gradient_dicts)
    return avg_grad

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=N_MULTIPROCESS)

    total_episodes = N_EPISODES
    episode_counter = 0

    total_sampling_time = 0.0
    total_learning_time = 0.0
    total_aggregation_time = 0.0

    env_main = JobRoutingGymEnv(Config())
    model = build_model(env_main)

    start_time = time.time()

    while episode_counter < total_episodes:
        batch_workers = min(N_MULTIPROCESS, total_episodes - episode_counter)
        model_state_dict = {k: v.cpu() for k, v in model.policy.state_dict().items()}
        tasks = [(i, model_state_dict, episode_counter + i + 1) for i in range(batch_workers)]

        results = pool.map(worker_wrapper, tasks)

        gradients_list = []
        for core_index, sampling_time, learn_time, avg_reward, gradients, ep_len, ep_idx in results:
            episode_counter += 1
            total_sampling_time += sampling_time
            total_learning_time += learn_time
            gradients_list.append(gradients)

            main_writer.add_scalar(f"reward_core_{core_index+1}", avg_reward, episode_counter)
            main_writer.add_scalar("reward_average", avg_reward, episode_counter)
            print(f"[Worker {core_index}] Episode {ep_idx}: "
                  f"Sampling {sampling_time:.6f}s, Learn {learn_time:.6f}s, "
                  f"Reward(avg) {avg_reward:.2f}")

        # 평균 gradient 적용
        start_agg = time.time()
        avg_grad = average_gradients(gradients_list)
        model.apply_gradients(avg_grad)
        aggregation_time = time.time() - start_agg
        total_aggregation_time += aggregation_time

    total_time = (time.time() - start_time) / 60
    total_sampling_time = total_sampling_time / N_MULTIPROCESS
    total_learning_time = total_learning_time / N_MULTIPROCESS

    print(f"\n[Experiment Summary] "
          f"Total Sampling {total_sampling_time:.6f}s | "
          f"Total Learning {total_learning_time:.6f}s | "
          f"Total Aggregation {total_aggregation_time:.6f}s | "
          f"Total Time {total_time:.6f}min\n")

    pool.close()
    pool.join()
"""
import os
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from environment import JobRoutingGymEnv, Config
from PPO import PPOAgent   # 네가 만든 PPO.py 안에 있는 PPOAgent

N_EPISODES = 1000

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
    ep_loss_sum, ep_loss_count = 0.0, 0   # loss 기록용

    for t in range(1, total_timesteps + 1):
        # PPO action
        action_vec, log_prob = agent.select_action(obs)
        action = int(action_vec[0])

        # step
        obs2, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # transition 저장
        agent.store_transition((obs, action_vec, reward, obs2, done, log_prob.item()))

        obs = obs2
        ep_ret += reward
        ep_len += 1

        # episode 종료
        if done:
            ep_count += 1
            avg_reward = ep_ret / ep_len

            # PPO 업데이트
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
            ep_loss_sum, ep_loss_count = 0.0, 0   # 초기화

    # 최종 평가
    mean_ret, std_ret = evaluate(agent, eval_env, n_episodes=20)
    print(f"[EVAL] Return mean={int(mean_ret):,} ± {int(std_ret):,}")

    # 모델 저장
    agent.save("runs/ppo_agent.pt")
    writer.close()


if __name__ == "__main__":
    main()
