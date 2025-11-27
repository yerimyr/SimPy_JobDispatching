import os
import time
import csv
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from environment import Config, JobRoutingGymEnv
from PPO_multilearning import PPOAgent

cfg = Config()
N_EPISODES        = 5000
MAX_STEPS_PER_EP  = cfg.NUM_JOBS
STATE_DIM         = cfg.STATE_DIM
N_ACTIONS         = cfg.N_ACTIONS

TB_DIR   = Path("runs/tb")
CKPT_PTH = Path("ppo_ckpt_multilearn.pt")

TB_DIR.mkdir(parents=True, exist_ok=True)

main_writer = SummaryWriter(log_dir=str(TB_DIR / "MULTILEARN_ABCTest"))


def build_model(device: str):
    model = PPOAgent(
        state_dim=STATE_DIM,
        action_dims=[N_ACTIONS],
        device=device,
    )
    return model

def simulation_worker(core_index, model_state_dict, mode, actor_device, learner_device):

    env = JobRoutingGymEnv(Config())

    actor_agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dims=[N_ACTIONS],
        device=actor_device,
    )
    actor_agent.policy.load_state_dict(model_state_dict)

    start_sampling = time.time()
    s, _info = env.reset()
    done = False
    traj = []
    total_reward = 0.0

    for _ in range(MAX_STEPS_PER_EP):
        a_vec, logp = actor_agent.select_action(s)
        a_scalar = int(a_vec[0] if isinstance(a_vec, (list, np.ndarray)) else a_vec)

        s2, r, done, trunc, info = env.step(a_scalar)

        traj.append(
            (
                s,
                np.array([a_scalar], dtype=np.int64),
                float(r),
                s2,
                float(done),
                float(logp.item()),
            )
        )
        total_reward += r
        s = s2

        if done:
            break

    sampling_time = time.time() - start_sampling

    start_update = time.time()

    if learner_device == actor_device and mode in ("A", "B"):
        learner_agent = actor_agent
    else:
        learner_agent = PPOAgent(
            state_dim=STATE_DIM,
            action_dims=[N_ACTIONS],
            device=learner_device,
        )
        learner_agent.policy.load_state_dict(model_state_dict)

    for tr in traj:
        learner_agent.store_transition(tr)
        
    grads = learner_agent.compute_gradients()
    learning_time = time.time() - start_update

    learner_agent.memory.clear()
    env.close()

    gradients_cpu = {k: v.detach().cpu() for k, v in grads.items()}

    return core_index, sampling_time, learning_time, total_reward, gradients_cpu


def worker_wrapper(args):
    return simulation_worker(*args)


def average_gradients(gradient_dicts):
    if len(gradient_dicts) == 0:
        return {}

    avg_grad = {}
    keys = gradient_dicts[0].keys()
    for k in keys:
        avg_grad[k] = sum(d[k] for d in gradient_dicts) / len(gradient_dicts)
    return avg_grad


def run_experiment(mode: str, n_workers: int):

    mode = mode.upper()
    if mode not in ("A", "B", "C"):
        raise ValueError("mode must be in {'A','B','C'}")

    cuda_available = torch.cuda.is_available()

    if mode == "A":
        actor_device = learner_device = train_device = ("cuda" if cuda_available else "cpu")
    elif mode == "B":
        actor_device = learner_device = train_device = "cpu"
    else:  # C
        actor_device = "cpu"
        learner_device = "cuda" if cuda_available else "cpu"
        train_device   = learner_device

    print(f"\n=== RUN: mode={mode}, workers={n_workers} ===")
    print(f"actor_device={actor_device}, learner_device={learner_device}, train_device={train_device}")

    pool = multiprocessing.Pool(processes=n_workers)

    episode_counter = 0
    total_sampling_time = 0.0  
    total_learning_time = 0.0  
    total_aggregation_time = 0.0

    model = build_model(train_device)
    start_time = time.time()

    try:
        while episode_counter < N_EPISODES:

            batch_workers = min(n_workers, N_EPISODES - episode_counter)

            state_dict = model.policy.state_dict()
            model_state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}

            tasks = [(i, model_state_dict, mode, actor_device, learner_device)
                     for i in range(batch_workers)]

            results = pool.map(worker_wrapper, tasks)

            gradients_list = []
            batch_rewards = []

            batch_sampling_times = []
            batch_learning_times = []

            for core_index, sampling_time, learn_time, reward, gradients in results:
                episode_counter += 1

                batch_sampling_times.append(sampling_time)
                batch_learning_times.append(learn_time)

                gradients_list.append(gradients)
                batch_rewards.append(reward)

                main_writer.add_scalar(
                    f"{mode}_w{n_workers}/core_{core_index+1}/reward",
                    reward,
                    episode_counter,
                )

                print(
                    f"[{mode}][Worker {core_index}] Episode {episode_counter}: "
                    f"Sampling {sampling_time:.6f}s | Learn {learn_time:.6f}s | Reward {reward:.2f}"
                )

            batch_sampling_wall = max(batch_sampling_times)
            batch_learning_wall = max(batch_learning_times)

            total_sampling_time += batch_sampling_wall
            total_learning_time += batch_learning_wall

            if batch_rewards:
                avg_reward = float(np.mean(batch_rewards))
                main_writer.add_scalar(
                    f"{mode}_w{n_workers}/reward_average",
                    avg_reward,
                    episode_counter,
                )
                print(f"[{mode}][Batch] Up to {episode_counter} | AvgReward {avg_reward:.2f}")

            start_agg = time.time()
            avg_grad_cpu = average_gradients(gradients_list)

            if train_device != "cpu":
                avg_grad = {k: v.to(train_device) for k, v in avg_grad_cpu.items()}
            else:
                avg_grad = avg_grad_cpu

            model.apply_gradients(avg_grad)

            agg_time = time.time() - start_agg
            total_aggregation_time += agg_time

        total_time_sec = time.time() - start_time
        total_time_min = total_time_sec / 60.0

        print(
            f"\n[Experiment Summary] mode={mode}, workers={n_workers}\n"
            f"  Total Episodes           : {episode_counter}\n"
            f"  Total Sampling Time(wall): {total_sampling_time:.6f}s\n"
            f"  Total Learning Time(wall): {total_learning_time:.6f}s\n"
            f"  Total Aggregation Time   : {total_aggregation_time:.6f}s\n"
            f"  Total Wall Time          : {total_time_sec:.6f}s ({total_time_min:.3f} min)\n"
        )

        torch.save({"policy": model.policy.state_dict()}, str(CKPT_PTH))
        print(f"[ckpt] saved → {CKPT_PTH}")

        return {
            "mode": mode,
            "workers": n_workers,
            "episodes": episode_counter,
            "total_sampling_time_s": total_sampling_time,
            "total_learning_time_s": total_learning_time,
            "total_aggregation_time_s": total_aggregation_time,
            "total_wall_time_s": total_time_sec,
        }

    finally:
        pool.close()
        pool.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    run_plan = [
        ("A", 1), ("A", 5),
        ("B", 1), ("B", 5),
        ("C", 1), ("C", 5),
    ]

    out_csv = Path("ppo_multilearn_ABCTest_summary.csv")
    fieldnames = [
        "mode", "workers", "episodes",
        "total_sampling_time_s",
        "total_learning_time_s",
        "total_aggregation_time_s",
        "total_wall_time_s",
    ]

    results = []

    for mode, workers in run_plan:
        res = run_experiment(mode, workers)
        results.append(res)

        write_header = not out_csv.exists()
        with out_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(res)

    print(f"\n[CSV] saved → {out_csv}\n")
    main_writer.close()
