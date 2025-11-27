# 멀티코어 비동기 DQN (A3C/IMPALA 스타일) - PPO 멀티러닝 코드 형식 유지 + Gantt 기반 wall time 측정 버전
import os
import time
import csv
import multiprocessing
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from environment import Config, JobRoutingGymEnv
from DQN_multilearning import DQNAgent   # ★ PPOAgent → DQNAgent


# ====== 기본 설정 ======
cfg = Config()
N_EPISODES        = 5000
MAX_STEPS_PER_EP  = cfg.NUM_JOBS
STATE_DIM         = cfg.STATE_DIM
N_ACTIONS         = cfg.N_ACTIONS

TB_DIR   = Path("runs/tb")                 # 형식 유지
CKPT_PTH = Path("dqn_ckpt_multilearn.pt")  # 이름만 DQN으로 살짝 변경

TB_DIR.mkdir(parents=True, exist_ok=True)

# TensorBoard (공용)
main_writer = SummaryWriter(log_dir=str(TB_DIR / "MULTILEARN_DQN_ASYNC_ABCTest"))


# ====== Gantt-style union time 계산 함수 ======
def _gantt_total_time(intervals):
    """
    intervals: [(start, end), ...]
    여러 워커가 병렬로 작업한 구간들의 union 길이 계산.
    예) [0,2], [1,3] → union = [0,3] → 3초
    """
    if not intervals:
        return 0.0

    # 시작시간 기준 정렬
    intervals = sorted(intervals, key=lambda x: x[0])

    merged = []
    cur_s, cur_e = intervals[0]

    for s, e in intervals[1:]:
        if s <= cur_e:
            # 겹치면 구간 병합
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # 병합된 구간들의 길이 합
    return sum(e - s for (s, e) in merged)


# ====== 메인 모델 빌드 (Global DQN 서버) ======
def build_model(device: str):
    model = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=N_ACTIONS,
        device=device,
    )
    return model


# ====== 워커: 샘플링 + 로컬 compute_gradients ======
def simulation_worker(args):
    """
    args: (core_index, model_state_dict, mode, actor_device, learner_device)
    - 한 워커가 '1개 에피소드'를 처리하고,
      샘플링 + 로컬 DQN gradient를 계산해서 바로 반환하는 역할.
    """
    core_index, model_state_dict, mode, actor_device, learner_device = args

    env = JobRoutingGymEnv(Config())

    # --- Actor DQN (sampling용) ---
    actor_agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=N_ACTIONS,
        device=actor_device,
    )
    # global model snapshot 로드 (q_network / target_network 둘 다)
    actor_agent.q_network.load_state_dict(model_state_dict)
    actor_agent.target_network.load_state_dict(model_state_dict)

    # --- 샘플링 ---
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

    # --- Learner DQN (gradient 계산용) ---
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

    # trajectory → replay buffer
    for (s, a, r, s2, done) in traj:
        learner_agent.replay_buffer.push(s, a, r, s2, done)

    # DQN gradient 계산 (replay buffer가 너무 작으면 skip)
    try:
        grads = learner_agent.compute_gradients()
    except ValueError:
        grads = {}

    t1_learn = time.time()
    learning_time = t1_learn - t0_learn

    env.close()

    gradients_cpu = {k: v.detach().cpu() for k, v in grads.items()}

    # 비동기 콜백에서 바로 global model에 적용할 수 있도록 반환
    return (
        core_index,
        sampling_time,
        learning_time,
        total_reward,
        gradients_cpu,
        t0_sam,
        t1_sam,
        t0_learn,
        t1_learn,
    )


def worker_wrapper(args):
    return simulation_worker(args)


# ====== 실험 실행 (비동기 gradient 적용 + Gantt 기반 wall time) ======
def run_experiment(mode: str, n_workers: int):

    mode = mode.upper()
    if mode not in ("A", "B", "C"):
        raise ValueError("mode must be in {'A','B','C'}")

    cuda_available = torch.cuda.is_available()

    # A/B/C 모드는 PPO 버전과 동일한 의미 유지
    if mode == "A":
        actor_device = learner_device = train_device = ("cuda" if cuda_available else "cpu")
    elif mode == "B":
        actor_device = learner_device = train_device = "cpu"
    else:  # C: actor=CPU, learner/server=GPU
        actor_device = "cpu"
        learner_device = "cuda" if cuda_available else "cpu"
        train_device   = learner_device

    print(f"\n=== RUN (DQN ASYNC): mode={mode}, workers={n_workers} ===")
    print(f"actor_device={actor_device}, learner_device={learner_device}, train_device={train_device}")

    # ----- Global DQN 서버 -----
    model = build_model(train_device)

    # 상태 관리 (동기 PPO 형식은 유지하되 내부는 async로 동작)
    episode_counter         = 0          # 완료된 episode 수
    episodes_issued         = 0          # 워커들에 던진 episode 수
    in_flight               = 0          # 현재 돌고 있는 워커 task 수

    # 아래 3개는 "합(sum)"이 아니라, interval union 기반으로 다시 계산할 것이므로
    # 여기서는 참고용으로만 쓰거나 사용 안 해도 됨.
    total_sampling_time_sum    = 0.0     # (옵션) 단순 합
    total_learning_time_sum    = 0.0     # (옵션) 단순 합
    total_aggregation_time     = 0.0     # apply_gradients 시간 합 (직렬이므로 sum이 wall time)

    # Gantt 기반 wall time 계산용 interval 리스트
    sampling_intervals = []  # [(t0_sam, t1_sam), ...]
    learning_intervals = []  # [(t0_learn, t1_learn), ...]

    start_time = time.time()
    pool = multiprocessing.Pool(processes=n_workers)

    # ----- 태스크 제출 -----
    def _submit_task(core_index: int):
        nonlocal episodes_issued, in_flight

        if episodes_issued >= N_EPISODES:
            return False

        # 최신 global q_network snapshot 가져오기
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

    # ----- 워커 수 유지 (비동기) -----
    def _maybe_issue_more():
        nonlocal in_flight, episodes_issued
        while (in_flight < n_workers) and (episodes_issued < N_EPISODES):
            core_idx = episodes_issued  # 단순 ID
            if not _submit_task(core_idx):
                break

    # ----- 콜백: 워커 하나가 끝날 때마다 즉시 global model 업데이트 -----
    def _on_result(res):
        nonlocal episode_counter, in_flight
        nonlocal total_sampling_time_sum, total_learning_time_sum, total_aggregation_time
        nonlocal sampling_intervals, learning_intervals

        (
            core_index,
            sampling_time,
            learning_time,
            reward,
            gradients_cpu,
            t0_sam,
            t1_sam,
            t0_learn,
            t1_learn,
        ) = res

        try:
            episode_counter += 1
            ep = episode_counter

            # (옵션) 단순 합 통계
            total_sampling_time_sum += sampling_time
            total_learning_time_sum += learning_time

            # Gantt union용 interval 기록
            sampling_intervals.append((t0_sam, t1_sam))
            learning_intervals.append((t0_learn, t1_learn))

            # TensorBoard에 episode별 reward 기록 (형식 유지)
            main_writer.add_scalar(
                f"{mode}_w{n_workers}/core_{core_index+1}/reward",
                reward,
                ep,
            )

            print(
                f"[{mode}][Worker {core_index}] Episode {ep}: "
                f"Sampling {sampling_time:.6f}s | Learn {learning_time:.6f}s | Reward {reward:.2f}"
            )

            # ---- 비동기 gradient 적용 (평균 없이, 들어오는 순서대로) ----
            if gradients_cpu:
                grads_device = {k: v.to(train_device) for k, v in gradients_cpu.items()}
                t0_agg = time.time()
                model.apply_gradients(grads_device)
                t1_agg = time.time()
                total_aggregation_time += (t1_agg - t0_agg)

            # "평균 reward" 채널은 단일 episode reward를 그대로 사용 (형식만 맞춤)
            main_writer.add_scalar(
                f"{mode}_w{n_workers}/reward_average",
                float(reward),
                ep,
            )

        finally:
            in_flight -= 1
            _maybe_issue_more()

    # ----- 에러 콜백 -----
    def _on_error(exc):
        nonlocal in_flight
        print("[Async Error]", exc)
        in_flight -= 1
        _maybe_issue_more()

    # 초기 태스크 발행
    _maybe_issue_more()

    # ===== 메인 루프: 모든 episode 끝날 때까지 대기 =====
    try:
        while (episode_counter < N_EPISODES) or (in_flight > 0):
            time.sleep(0.01)

        # ===== 전체 요약 =====
        total_time_sec = time.time() - start_time
        total_time_min = total_time_sec / 60.0

        # Gantt 기반 병렬 wall time 계산
        sampling_wall  = _gantt_total_time(sampling_intervals)
        learning_wall  = _gantt_total_time(learning_intervals)
        aggregation_wall = total_aggregation_time  # apply_gradients는 직렬 실행이므로 sum = wall

        print(
            f"\n[Experiment Summary - DQN ASYNC] mode={mode}, workers={n_workers}\n"
            f"  Total Episodes                : {episode_counter}\n"
            f"  (Sum) Sampling Time           : {total_sampling_time_sum:.6f}s\n"
            f"  (Sum) Learning Time           : {total_learning_time_sum:.6f}s\n"
            f"  (Gantt) Sampling Time (wall)  : {sampling_wall:.6f}s\n"
            f"  (Gantt) Learning Time (wall)  : {learning_wall:.6f}s\n"
            f"  Aggregation Time (sum & wall) : {aggregation_wall:.6f}s\n"
            f"  Total Wall Time               : {total_time_sec:.6f}s ({total_time_min:.3f} min)\n"
        )

        torch.save({"policy": model.q_network.state_dict()}, str(CKPT_PTH))
        print(f"[ckpt] saved → {CKPT_PTH}")

        # CSV에는 Gantt 기반 wall time을 기록
        return {
            "mode": mode,
            "workers": n_workers,
            "episodes": episode_counter,
            "total_sampling_time_s": sampling_wall,
            "total_learning_time_s": learning_wall,
            "total_aggregation_time_s": aggregation_wall,
            "total_wall_time_s": total_time_sec,
        }

    finally:
        pool.close()
        pool.join()


# ====== Main ======
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
