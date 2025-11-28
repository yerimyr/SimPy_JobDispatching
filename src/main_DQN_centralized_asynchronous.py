import os
import time
import csv
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from environment import Config, JobRoutingGymEnv
from DQN_multilearning import DQNAgent   


cfg = Config()
N_EPISODES        = 5000
MAX_STEPS_PER_EP  = cfg.NUM_JOBS
STATE_DIM         = cfg.STATE_DIM
N_ACTIONS         = cfg.N_ACTIONS

TB_DIR   = Path("runs/tb")                 
CKPT_PTH = Path("dqn_ckpt_multilearn.pt")  
TB_DIR.mkdir(parents=True, exist_ok=True)

main_writer = SummaryWriter(log_dir=str(TB_DIR / "MULTILEARN_DQN_ASYNC_ABCTest"))


def build_model(device: str):
    model = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=N_ACTIONS,
        device=device,
    )
    return model


def simulation_worker(args):
    core_index, model_state_dict, mode, actor_device, learner_device = args

    env = JobRoutingGymEnv(Config())

    actor_agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=N_ACTIONS,
        device=actor_device,
    )

    actor_agent.q_network.load_state_dict(model_state_dict)
    actor_agent.target_network.load_state_dict(model_state_dict)

    t0_sam = time.time()
    s, _info = env.reset()
    traj = []
    total_reward = 0.0

    for _ in range(MAX_STEPS_PER_EP):
        a = actor_agent.select_action(s)
        s2, r, done, trunc, info = env.step(a)

        traj.append((s, a, r, s2, float(done)))
        total_reward += r
        s = s2

        if done:
            break

    t1_sam = time.time()
    sampling_time = t1_sam - t0_sam

    t0_learn = time.time()

    if learner_device == actor_device and mode in ("A", "B"):
        learner_agent = actor_agent
    else:
        learner_agent = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=N_ACTIONS,
            device=learner_device,
        )
        learner_agent.q_network.load_state_dict(model_state_dict)
        learner_agent.target_network.load_state_dict(model_state_dict)

    for (s, a, r, s2, done) in traj:
        learner_agent.replay_buffer.push(s, a, r, s2, done)

    grads = {}
    len_buf = len(learner_agent.replay_buffer)

    if len_buf > 0:
        orig_min_buf = learner_agent.min_buffer_size
        orig_bs      = learner_agent.batch_size

        learner_agent.min_buffer_size = 1
        learner_agent.batch_size      = min(orig_bs, len_buf)

        try:
            grads = learner_agent.compute_gradients()
        except ValueError:
            grads = {}
        finally:
            learner_agent.min_buffer_size = orig_min_buf
            learner_agent.batch_size      = orig_bs

    t1_learn = time.time()
    learning_time = t1_learn - t0_learn

    env.close()

    gradients_cpu = {k: v.detach().cpu() for k, v in grads.items()}

    return (
        core_index,
        sampling_time,
        learning_time,
        total_reward,
        gradients_cpu,
        t1_learn,   
    )


def worker_wrapper(args):
    return simulation_worker(args)


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

    print(f"\n=== RUN (DQN ASYNC): mode={mode}, workers={n_workers} ===")
    print(f"actor_device={actor_device}, learner_device={learner_device}, train_device={train_device}")

    model = build_model(train_device)

    episode_counter         = 0          
    episodes_issued         = 0          
    in_flight               = 0          

    cycle_sampling_sum   = 0.0
    cycle_learning_sum   = 0.0
    cycle_waiting_sum    = 0.0
    cycle_agg_sum        = 0.0
    cycle_count          = 0           

    total_sampling_time_s     = 0.0
    total_learning_time_s     = 0.0
    total_waiting_time_s      = 0.0
    total_aggregation_time_s  = 0.0

    start_time = time.time()
    pool = multiprocessing.Pool(processes=n_workers)

    def _submit_task(core_index: int):
        nonlocal episodes_issued, in_flight

        if episodes_issued >= N_EPISODES:
            return False

        state_dict = {k: v.detach().cpu() for k, v in model.q_network.state_dict().items()}

        args = (core_index, state_dict, mode, actor_device, learner_device)

        pool.apply_async(
            worker_wrapper,
            args=(args,),
            callback=_on_result,
            error_callback=_on_error,
        )

        episodes_issued += 1
        in_flight       += 1
        return True

    def _maybe_issue_more():
        nonlocal in_flight, episodes_issued
        while (in_flight < n_workers) and (episodes_issued < N_EPISODES):
            core_idx = episodes_issued  
            if not _submit_task(core_idx):
                break

    def _on_result(res):
        nonlocal episode_counter, in_flight
        nonlocal cycle_sampling_sum, cycle_learning_sum, cycle_waiting_sum, cycle_agg_sum, cycle_count
        nonlocal total_sampling_time_s, total_learning_time_s, total_waiting_time_s, total_aggregation_time_s

        (
            core_index,
            sampling_time,
            learning_time,
            reward,
            gradients_cpu,
            t_end_compute,   
        ) = res

        try:
            episode_counter += 1
            ep = episode_counter

            main_writer.add_scalar(
                f"{mode}_w{n_workers}/core_{core_index+1}/reward",
                reward,
                ep,
            )

            print(
                f"[{mode}][Worker {core_index}] Episode {ep}: "
                f"Sampling {sampling_time:.6f}s | Learn {learning_time:.6f}s | Reward {reward:.2f}"
            )

            waiting_time = 0.0
            agg_time     = 0.0

            if gradients_cpu:
                t0_agg = time.time()
                waiting_time = t0_agg - t_end_compute

                grads_device = {k: v.to(train_device) for k, v in gradients_cpu.items()}
                model.apply_gradients(grads_device)

                t1_agg = time.time()
                agg_time = t1_agg - t0_agg

            cycle_sampling_sum += sampling_time
            cycle_learning_sum += learning_time
            cycle_waiting_sum  += waiting_time
            cycle_agg_sum      += agg_time
            cycle_count        += 1

            main_writer.add_scalar(
                f"{mode}_w{n_workers}/reward_average",
                float(reward),
                ep,
            )

            if (cycle_count == n_workers) or (episode_counter == N_EPISODES):
                denom = float(n_workers)  
                cycle_sampling_avg = cycle_sampling_sum / denom
                cycle_learning_avg = cycle_learning_sum / denom
                cycle_waiting_avg  = cycle_waiting_sum  / denom
                cycle_agg_sum_val  = cycle_agg_sum      

                total_sampling_time_s    += cycle_sampling_avg
                total_learning_time_s    += cycle_learning_avg
                total_waiting_time_s     += cycle_waiting_avg
                total_aggregation_time_s += cycle_agg_sum_val

                cycle_sampling_sum = 0.0
                cycle_learning_sum = 0.0
                cycle_waiting_sum  = 0.0
                cycle_agg_sum      = 0.0
                cycle_count        = 0

        finally:
            in_flight -= 1
            _maybe_issue_more()

    def _on_error(exc):
        nonlocal in_flight
        print("[Async Error]", exc)
        in_flight -= 1
        _maybe_issue_more()

    _maybe_issue_more()

    try:
        while (episode_counter < N_EPISODES) or (in_flight > 0):
            time.sleep(0.01)

        total_time_sec = time.time() - start_time
        total_time_min = total_time_sec / 60.0

        print(
            f"\n[Experiment Summary - DQN ASYNC] mode={mode}, workers={n_workers}\n"
            f"  Total Episodes                : {episode_counter}\n"
            f"  Total Sampling Time (cycles)  : {total_sampling_time_s:.6f}s\n"
            f"  Total Learning Time (cycles)  : {total_learning_time_s:.6f}s\n"
            f"  Total Waiting Time  (cycles)  : {total_waiting_time_s:.6f}s\n"
            f"  Total Aggregation Time (sum)  : {total_aggregation_time_s:.6f}s\n"
            f"  Total Wall Time               : {total_time_sec:.6f}s ({total_time_min:.3f} min)\n"
        )

        torch.save({"policy": model.q_network.state_dict()}, str(CKPT_PTH))
        print(f"[ckpt] saved → {CKPT_PTH}")

        return {
            "mode": mode,
            "workers": n_workers,
            "episodes": episode_counter,
            "total_sampling_time_s": total_sampling_time_s,
            "total_learning_time_s": total_learning_time_s,
            "total_waiting_time_s": total_waiting_time_s,
            "total_aggregation_time_s": total_aggregation_time_s,
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

    out_csv = Path("dqn_multilearn_async_ABCTest_summary.csv")
    fieldnames = [
        "mode",
        "workers",
        "episodes",
        "total_sampling_time_s",
        "total_learning_time_s",
        "total_waiting_time_s",
        "total_aggregation_time_s",
        "total_wall_time_s",
    ]

    results = []

    for mode, workers in run_plan:
        print(f"\n### RUN start: mode={mode}, workers={workers} (DQN ASYNC) ###\n")
        res = run_experiment(mode, workers)
        results.append(res)

        write_header = not out_csv.exists()
        with out_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: res.get(k) for k in fieldnames})

    print(f"\n[CSV] saved/updated → {out_csv}\n")
    main_writer.close()
