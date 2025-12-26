import os, io, time, csv
from pathlib import Path
from multiprocessing import Pool, get_start_method
import torch
import numpy as np

from environment import Config, JobRoutingGymEnv
from DQN import DQNAgent   


cfg = Config()
N_EPISODES        = 5000
MAX_STEPS_PER_EP  = cfg.NUM_JOBS
STATE_DIM         = cfg.STATE_DIM
N_ACTIONS         = cfg.N_ACTIONS

TB_DIR   = Path("runs/tb")
CKPT_PTH = Path("dqn_ckpt.pt")

EPISODES_PER_WORKER = 1
SAVE_EVERY = 10


def _serialize_q(agent) -> bytes:
    bio = io.BytesIO()
    torch.save(agent.q_network.state_dict(), bio)
    return bio.getvalue()

def _load_q_weights(agent, blob: bytes, device_str: str):
    if not blob:
        return
    sd = torch.load(io.BytesIO(blob), map_location=torch.device(device_str))
    agent.q_network.load_state_dict(sd)
    agent.target_network.load_state_dict(sd)


def _worker_rollout(args):
    wid, weights_blob, episodes, max_steps, inference_device = args

    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=N_ACTIONS,
        device=inference_device
    )
    _load_q_weights(agent, weights_blob, inference_device)

    env = JobRoutingGymEnv(Config())

    episodes_traj = []
    ep_sample_times = []

    for _ in range(episodes):
        s, _ = env.reset()
        traj = []
        ep_return = 0.0
        t0 = time.time()

        for _t in range(max_steps):
            a = agent.select_action(s)
            s2, r, done, _, _ = env.step(a)
            traj.append((s, a, r, s2, float(done)))
            ep_return += r
            s = s2
            if done:
                break

        episodes_traj.append((traj, ep_return, s2))
        ep_sample_times.append(time.time() - t0)

    env.close()
    return episodes_traj, ep_sample_times


def _collect_async(pool, weights_blob, n_workers, episodes_per_worker,
                   max_steps, inference_device):
    tasks = [
        pool.apply_async(
            _worker_rollout,
            args=((wid, weights_blob, episodes_per_worker,
                   max_steps, inference_device),)
        )
        for wid in range(n_workers)
    ]

    all_eps, all_times = [], []

    for t in tasks:   
        trajs, times = t.get()
        all_eps.extend(trajs)
        all_times.extend(times)

    return all_eps, all_times


def main(mode="C", workers=5):
    mode = mode.upper()
    infer_dev = "cuda" if mode == "A" else "cpu"
    train_dev = "cuda" if mode in ("A", "C") else "cpu"

    print(f"\n=== RUN DQN ASYNC mode={mode}, workers={workers} ===")
    print(f"infer_device={infer_dev}, train_device={train_dev}")

    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=N_ACTIONS,
        device=train_dev
    )

    pool = Pool(processes=int(workers))

    ep = 0
    t0_total = time.time()
    sampling_time_acc = 0.0
    learning_time_acc = 0.0

    try:
        while ep < N_EPISODES:
            weights_blob = _serialize_q(agent)

            # ---------- sampling ----------
            t0 = time.time()
            episodes_traj, ep_times = _collect_async(
                pool,
                weights_blob,
                workers,
                EPISODES_PER_WORKER,
                MAX_STEPS_PER_EP,
                infer_dev
            )
            sampling_time_acc += time.time() - t0

            # ---------- learning ----------
            t1 = time.time()
            for (traj, ep_return, last_state), _st in zip(episodes_traj, ep_times):
                for (s, a, r, s2, d) in traj:
                    agent.replay_buffer.push(s, a, r, s2, d)

                loss = agent.update()
                ep += 1

                print(f"EP {ep:05d} | Return={ep_return:.3f} | Loss={loss}")

            learning_time_acc += time.time() - t1

            if SAVE_EVERY and ep % SAVE_EVERY == 0:
                torch.save(agent.q_network.state_dict(), CKPT_PTH)
                print(f"[ckpt] saved → {CKPT_PTH}")

    finally:
        pool.close()
        pool.join()

    total_time = time.time() - t0_total

    print("\n===== SUMMARY =====")
    print(f"Sampling Time : {sampling_time_acc:.1f}s")
    print(f"Learning Time : {learning_time_acc:.1f}s")
    print(f"Total Time    : {total_time:.1f}s")
    print("===================")

    return {
        "mode": mode,
        "workers": int(workers),
        "episodes": int(ep),
        "sampling_time_s": float(sampling_time_acc),
        "learning_time_s": float(learning_time_acc),
        "total_time_s": float(total_time),
        "finished_at_ts": float(time.time()),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "ckpt_path": str(CKPT_PTH),
    }


if __name__ == "__main__":
    if get_start_method(allow_none=True) != "spawn":
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)

    here = Path(__file__).resolve().parent
    out_csv = here.parent / "dqn_async_run_summary.csv"

    fieldnames = [
        "mode",
        "workers",
        "episodes",
        "sampling_time_s",
        "learning_time_s",
        "total_time_s",
        "finished_at_ts",
        "finished_at",
        "ckpt_path",
    ]

    for mode, w in [("A",1),("A",5),("B",1),("B",5),("C",1),("C",5)]:
        print(f"\n### RUN start: mode={mode}, workers={w} ###\n")
        res = main(mode, w)

        write_header = not out_csv.exists()
        with out_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: res.get(k) for k in fieldnames})

    print(f"\n[CSV] saved/updated → {out_csv}\n")
