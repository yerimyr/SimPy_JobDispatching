import os, io, time
from pathlib import Path
from multiprocessing import Pool, get_start_method
import torch
import numpy as np
from environment import Config, JobRoutingGymEnv
from PPO import PPOAgent


cfg = Config()
N_EPISODES        = 5000    
MAX_STEPS_PER_EP  = cfg.NUM_JOBS   
STATE_DIM         = cfg.STATE_DIM
N_ACTIONS         = cfg.N_ACTIONS

TB_DIR   = Path("runs/tb")           
CKPT_PTH = Path("ppo_ckpt.pt")    

EPISODES_PER_WORKER = 1  
SAVE_EVERY = 10          


def _serialize_policy(agent) -> bytes:
    bio = io.BytesIO()
    torch.save({"policy": agent.policy.state_dict()}, bio)
    return bio.getvalue()

def _load_policy_weights(agent, blob: bytes, device_str: str):
    if not blob:
        return
    sd = torch.load(io.BytesIO(blob), map_location=torch.device(device_str))
    agent.policy.load_state_dict(sd["policy"])

def _worker_rollout(args):
    wid, weights_blob, episodes, max_steps, inference_device = args

    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dims=[N_ACTIONS],
        device=("cuda" if inference_device == "cuda" else "cpu")
    )
    _load_policy_weights(agent, weights_blob,
                         device_str=("cuda" if inference_device == "cuda" else "cpu"))

    cfg = Config()
    env = JobRoutingGymEnv(cfg)

    episodes_traj = []
    ep_sample_times = []

    for _ in range(episodes):
        s, _info = env.reset()
        ep_traj = []
        cur_return = 0.0
        t0_ep = time.time()

        for _t in range(max_steps):
            a_vec, logp = agent.select_action(s)
            a_scalar = int(a_vec[0]) if isinstance(a_vec, (list, np.ndarray)) else int(a_vec)
            s2, r, done, _trunc, _i = env.step(a_scalar)
            ep_traj.append((s, np.array([a_scalar], dtype=np.int64),
                            float(r), s2, float(done), float(logp.item())))
            cur_return += r
            s = s2
            if done:
                break

        episodes_traj.append((ep_traj, cur_return, s2))  
        ep_sample_times.append(time.time() - t0_ep)

    env.close()
    return episodes_traj, ep_sample_times


def _collect_transitions_with_pool(pool, weights_blob, n_workers, episodes_per_worker, max_steps, inference_device):
    tasks = [(wid, weights_blob, episodes_per_worker, max_steps, inference_device)
             for wid in range(n_workers)]
    if n_workers == 1:
        return _worker_rollout(tasks[0])
    outs = pool.map(_worker_rollout, tasks)
    all_eps, ep_times_all = [], []
    for traj, times in outs:
        all_eps.extend(traj)
        ep_times_all.extend(times)
    return all_eps, ep_times_all


def _build_trainer(train_device: str):
    agent = PPOAgent(state_dim=STATE_DIM, action_dims=[N_ACTIONS], device=train_device)
    return agent


def _train_on_batch(trainer, episodes_traj, ep_times, writer, ep_counter):
    for (ep_traj, ep_return, last_state), sample_time in zip(episodes_traj, ep_times):
        for (s, a_vec, r, s2, d, logp) in ep_traj:
            trainer.store_transition((s, a_vec, r, s2, d, logp))

        ep_counter += 1
        if writer is not None:
            writer.add_scalar("Return/train", float(ep_return), ep_counter)

        st_str = f"{sample_time:.3f}s" if sample_time is not None else "N/A"
        sim_raw = "N/A" if last_state is None else str(last_state[-1])  
        print(f"EP {ep_counter:05d} | Return={ep_return:.3f} | SimEnd={sim_raw} | SampleTime={st_str}")

    learn_time = trainer.update()
    if writer is not None and learn_time is not None:
        writer.add_scalar("Time/LearnSec", float(learn_time), ep_counter)

    return ep_counter


def main(mode: str = "C", workers: int = 5):
    mode = (mode or "C").upper()
    if mode not in ("A","B","C"):
        raise ValueError("mode must be one of {'A','B','C'}")

    infer_dev = "cuda" if mode == "A" else "cpu"
    train_dev = "cuda" if mode in ("A","C") else "cpu"

    run_name = f"{mode}_w{int(workers)}"
    log_dir  = TB_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(log_dir))

    trainer = _build_trainer(train_dev)

    t0_total = time.time()
    sampling_time_acc = 0.0
    learning_time_acc = 0.0

    ep = 0

    pool = Pool(processes=int(workers))
    try:
        while ep < N_EPISODES:
            blob = _serialize_policy(trainer)

            t0_sam = time.time()
            episodes_traj, ep_times = _collect_transitions_with_pool(
                pool=pool,
                weights_blob=blob,
                n_workers=int(workers),
                episodes_per_worker=EPISODES_PER_WORKER,
                max_steps=MAX_STEPS_PER_EP,
                inference_device=infer_dev
            )
            t1_sam = time.time()
            sampling_time_acc += (t1_sam - t0_sam)

            t0_learn = time.time()
            ep = _train_on_batch(trainer, episodes_traj, ep_times, writer, ep)
            t1_learn = time.time()
            learning_time_acc += (t1_learn - t0_learn)

            if SAVE_EVERY and (ep % SAVE_EVERY == 0):
                torch.save({"policy": trainer.policy.state_dict()}, str(CKPT_PTH))
                print(f"[ckpt] saved → {CKPT_PTH}")

        torch.save({"policy": trainer.policy.state_dict()}, str(CKPT_PTH))
        t1_total = time.time()
        total_time = (t1_total - t0_total)
        print("\n===== SUMMARY =====")
        print(f"Sampling Time : {sampling_time_acc:.1f}s")
        print(f"Learning Time : {learning_time_acc:.1f}s")
        print(f"Total Time    : {total_time:.1f}s")
        print(f"Finished at   : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1_total))}")
        print("===================\n")

        return {
            "mode": mode,
            "workers": int(workers),
            "episodes": int(ep),
            "sampling_time_s": float(sampling_time_acc),
            "learning_time_s": float(learning_time_acc),
            "total_time_s": float(total_time),
            "finished_at_ts": float(t1_total),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1_total)),
            "tb_log_dir": str(log_dir),
        }
    finally:
        pool.close()
        pool.join()
        writer.close()


if __name__ == "__main__":
    if get_start_method(allow_none=True) != "spawn":
        import multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    run_plan = [
        ("A", 1),
        ("A", 5),
        ("B", 1),
        ("B", 5),
        ("C", 1),
        ("C", 5),
    ]

    here = Path(__file__).resolve().parent
    out_csv = here.parent / "ppo_run_summary.csv"

    import csv
    fieldnames = [
        "mode", "workers", "episodes",
        "sampling_time_s", "learning_time_s", "total_time_s",
        "finished_at_ts", "finished_at", "tb_log_dir"
    ]

    results = []
    for m, w in run_plan:
        print(f"\n### RUN start: mode={m}, workers={w} ###\n")
        res = main(m, workers=w)
        if res is None:
            res = {"mode": m, "workers": w, "episodes": 0,
                   "sampling_time_s": 0.0, "learning_time_s": 0.0,
                   "total_time_s": 0.0, "finished_at_ts": time.time(),
                   "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                   "tb_log_dir": str(TB_DIR / f"{m}_w{int(w)}")}
        results.append(res)

        write_header = not out_csv.exists()
        with out_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: res.get(k) for k in fieldnames})

    print(f"\n[CSV] saved/updated → {out_csv}\n")
