import os, io, time, csv
from pathlib import Path
from multiprocessing import Pool, get_start_method
import numpy as np
import torch

from environment import Config, JobRoutingGymEnv
from PPO_multilearning import PPOAgent

# ====== 설정 ======
cfg = Config()
N_EPISODES        = 5000
MAX_STEPS_PER_EP  = cfg.NUM_JOBS
STATE_DIM         = cfg.STATE_DIM
N_ACTIONS         = cfg.N_ACTIONS

TB_DIR   = Path("runs/tb")
CKPT_PTH = Path("ppo_ckpt.pt")

EPISODES_PER_WORKER = 1          # 워커당 에피소드 수
SAVE_EVERY = 10                  # 체크포인트 주기


# ====== 가중치 직렬화/역직렬화 ======
def _serialize_policy(agent) -> bytes:
    bio = io.BytesIO()
    torch.save({"policy": agent.policy.state_dict()}, bio)
    return bio.getvalue()

def _load_policy_weights(agent, blob: bytes, device_str: str):
    if not blob:
        return
    sd = torch.load(io.BytesIO(blob), map_location=torch.device(device_str))
    agent.policy.load_state_dict(sd["policy"])


# ====== 워커: 샘플링 + 로컬 학습(그라디언트 추출) ======
def _worker_rollout(args):
    """
    각 워커는 다음을 수행:
      1) 전달받은 전역 policy 파라미터 로드
      2) 에피소드 샘플링
      3) 메모리에 저장한 뒤 compute_gradients() 호출 → 그라디언트 딕셔너리 반환
    """
    wid, weights_blob, episodes, max_steps, device_str = args

    # 워커용 에이전트 (inference/learning 동일 디바이스)
    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dims=[N_ACTIONS],
        device=("cuda" if device_str == "cuda" else "cpu")
    )
    _load_policy_weights(agent, weights_blob, device_str=("cuda" if device_str == "cuda" else "cpu"))

    env = JobRoutingGymEnv(Config())

    all_grads = []          # 에피소드별 gradient dict
    ep_rewards = []         # 에피소드 리턴 모음
    sample_times = []       # 샘플링 시간
    learn_times = []        # 로컬 학습(그라디언트 계산) 시간

    for _ in range(episodes):
        s, _info = env.reset()
        traj = []
        ep_return = 0.0

        t0_sam = time.time()
        for _t in range(max_steps):
            a_vec, logp = agent.select_action(s)
            a_scalar = int(a_vec[0]) if isinstance(a_vec, (list, np.ndarray)) else int(a_vec)
            s2, r, done, _trunc, _i = env.step(a_scalar)
            traj.append((s, np.array([a_scalar], dtype=np.int64), float(r), s2, float(done), float(logp.item())))
            ep_return += r
            s = s2
            if done:
                break
        t1_sam = time.time()

        # 메모리에 쌓고 → 그라디언트 계산
        for tr in traj:
            agent.store_transition(tr)

        t0_learn = time.time()
        grads = agent.compute_gradients()
        t1_learn = time.time()

        # 워커 측에서는 메모리 비워서 다음 에피소드 대비
        agent.memory.clear()

        # CPU 텐서로 변환(IPC 안정성)
        grads_cpu = {k: v.detach().cpu() for k, v in grads.items()}

        all_grads.append(grads_cpu)
        ep_rewards.append(ep_return)
        sample_times.append(t1_sam - t0_sam)
        learn_times.append(t1_learn - t0_learn)

    env.close()
    return wid, all_grads, ep_rewards, sample_times, learn_times


# ====== Pool 재활용 수집 ======
def _collect_with_pool(pool, weights_blob, n_workers, episodes_per_worker, max_steps, device_str):
    tasks = [(wid, weights_blob, episodes_per_worker, max_steps, device_str) for wid in range(n_workers)]
    if n_workers == 1:
        return [_worker_rollout(tasks[0])]
    return pool.map(_worker_rollout, tasks)


# ====== 트레이너 빌드 ======
def _build_trainer(train_device: str):
    agent = PPOAgent(state_dim=STATE_DIM, action_dims=[N_ACTIONS], device=train_device)
    return agent


# ====== 평균 그라디언트 ======
def _average_gradients(grad_lists):
    """
    grad_lists: [grad_dict_episode1, grad_dict_episode2, ...] across all workers
    """
    if len(grad_lists) == 0:
        return {}

    # 모든 키 수집
    keys = set()
    for gd in grad_lists:
        keys.update(gd.keys())

    avg = {}
    for k in keys:
        terms = [gd[k] for gd in grad_lists if k in gd]
        if len(terms) == 0:
            continue
        # 동일 shape 보장됨
        stacked = torch.stack(terms, dim=0)
        avg[k] = stacked.mean(dim=0)
    return avg


# ====== 배치 결과를 TB/로그에 반영 ======
def _log_batch(writer, results, ep_counter):
    """
    results: list of (wid, grads_list, rewards_list, sample_times, learn_times)
    """
    for wid, _glist, rewards, s_times, l_times in results:
        # 워커별 평균 로깅
        avg_r = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
        avg_s = float(np.mean(s_times)) if len(s_times) > 0 else 0.0
        avg_l = float(np.mean(l_times)) if len(l_times) > 0 else 0.0
        print(f"[Worker {wid}] avg_return={avg_r:.3f} | sample={avg_s:.3f}s | learn={avg_l:.3f}s")

        if writer is not None:
            # 에피소드 카운터는 메인에서 관리(아래 main 루프에서 증가)
            writer.add_scalar(f"Worker{wid}/AvgReturn", avg_r, ep_counter)
            writer.add_scalar(f"Worker{wid}/AvgSampleSec", avg_s, ep_counter)
            writer.add_scalar(f"Worker{wid}/AvgLearnSec", avg_l, ep_counter)


# ====== 메인 ======
def main(mode: str = "C", workers: int = 5):
    """
    mode:
      - 'A': 샘플링/학습 모두 CUDA (가능한 경우)
      - 'B': 샘플링/학습 모두 CPU
      - 'C': 샘플링 CPU, 학습 CUDA (기본)
    """
    mode = (mode or "C").upper()
    if mode not in ("A", "B", "C"):
        raise ValueError("mode must be one of {'A','B','C'}")

    infer_dev = "cuda" if mode == "A" else ("cpu" if mode == "B" else "cpu")
    train_dev = "cuda" if mode in ("A", "C") and torch.cuda.is_available() else "cpu"

    run_name = f"{mode}_w{int(workers)}_PARALLEL_LEARN"
    log_dir = TB_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(log_dir))

    trainer = _build_trainer(train_dev)

    t0_total = time.time()
    sampling_time_acc = 0.0
    aggregation_time_acc = 0.0
    apply_time_acc = 0.0

    ep = 0

    pool = Pool(processes=int(workers))
    try:
        while ep < N_EPISODES:
            # 1) 전역 파라미터 브로드캐스트
            blob = _serialize_policy(trainer)

            # 2) 워커에서 샘플링 + 로컬 compute_gradients
            t0_sam = time.time()
            results = _collect_with_pool(
                pool=pool,
                weights_blob=blob,
                n_workers=int(workers),
                episodes_per_worker=EPISODES_PER_WORKER,
                max_steps=MAX_STEPS_PER_EP,
                device_str=infer_dev
            )
            t1_sam = time.time()
            sampling_time_acc += (t1_sam - t0_sam)

            # 3) 로그
            ep += int(workers) * EPISODES_PER_WORKER
            _log_batch(writer, results, ep_counter=ep)

            # 4) 모든 에피소드의 gradient 모아 평균
            t0_agg = time.time()
            flat_grads = []
            for _wid, grad_list, _rewards, _s_times, _l_times in results:
                # grad_list: [ep1_grads_dict, ep2_grads_dict, ...]
                flat_grads.extend(grad_list)
            avg_grad = _average_gradients(flat_grads)
            t1_agg = time.time()
            aggregation_time_acc += (t1_agg - t0_agg)

            # 5) 전역 모델에 평균 그래디언트 적용
            t0_apply = time.time()
            trainer.apply_gradients(avg_grad)
            t1_apply = time.time()
            apply_time_acc += (t1_apply - t0_apply)

            # 6) 체크포인트
            if SAVE_EVERY and (ep % SAVE_EVERY == 0):
                torch.save({"policy": trainer.policy.state_dict()}, str(CKPT_PTH))
                print(f"[ckpt] saved → {CKPT_PTH}")

        # 최종 저장 및 요약
        torch.save({"policy": trainer.policy.state_dict()}, str(CKPT_PTH))
        t1_total = time.time()
        total_time = (t1_total - t0_total)

        print("\n===== SUMMARY (Parallel Sampling + Parallel Learning) =====")
        print(f"Sampling Time : {sampling_time_acc:.1f}s")
        print(f"Aggregation   : {aggregation_time_acc:.1f}s")
        print(f"Apply Step    : {apply_time_acc:.1f}s")
        print(f"Total Time    : {total_time:.1f}s")
        print(f"Finished at   : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1_total))}")
        print("===========================================================\n")

        return {
            "mode": mode,
            "workers": int(workers),
            "episodes": int(ep),
            "sampling_time_s": float(sampling_time_acc),
            "aggregation_time_s": float(aggregation_time_acc),
            "apply_time_s": float(apply_time_acc),
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
    # Windows/macOS 호환 spawn
    if get_start_method(allow_none=True) != "spawn":
        import multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # 간단 실행 계획
    run_plan = [
        ("C", 1),
        ("C", 5),
        ("A", 1),
        ("A", 5),
        ("B", 1),
        ("B", 5),
    ]

    out_csv = Path("ppo_run_summary_parallel.csv")
    fieldnames = [
        "mode", "workers", "episodes",
        "sampling_time_s", "aggregation_time_s", "apply_time_s", "total_time_s",
        "finished_at_ts", "finished_at", "tb_log_dir"
    ]

    results = []
    for m, w in run_plan:
        print(f"\n### RUN start: mode={m}, workers={w} ###\n")
        res = main(m, workers=w)
        if res is None:
            res = {"mode": m, "workers": w, "episodes": 0,
                   "sampling_time_s": 0.0, "aggregation_time_s": 0.0, "apply_time_s": 0.0,
                   "total_time_s": 0.0, "finished_at_ts": time.time(),
                   "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                   "tb_log_dir": str(TB_DIR / f"{m}_w{int(w)}_PARALLEL_LEARN")}
        results.append(res)

        write_header = not out_csv.exists()
        with out_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: res.get(k) for k in fieldnames})

    print(f"\n[CSV] saved/updated → {out_csv}\n")
