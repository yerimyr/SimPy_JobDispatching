# main_DQN_centralized_asynchronous.py
import os, io, time, csv, traceback
from pathlib import Path
from multiprocessing import Pool, get_start_method
from collections import defaultdict

import numpy as np
import torch

from environment import Config, JobRoutingGymEnv
from DQN_multilearning import DQNAgent   # compute_gradients / apply_gradients 버전

# ====== 설정 ======
cfg = Config()
N_EPISODES        = 5000
MAX_STEPS_PER_EP  = cfg.NUM_JOBS
STATE_DIM         = cfg.STATE_DIM
N_ACTIONS         = cfg.N_ACTIONS

TB_DIR   = Path("runs_dqn_async/tb")
CKPT_PTH = Path("dqn_async_ckpt.pt")

EPISODES_PER_TASK = 1          # 태스크당 에피소드 수
SAVE_EVERY        = 200        # 체크포인트 저장 간격


# ====== 가중치 직렬화/역직렬화 ======
def _serialize_policy(agent) -> bytes:
    bio = io.BytesIO()
    torch.save({"policy": agent.q_network.state_dict()}, bio)
    return bio.getvalue()


def _load_policy_weights(agent, blob: bytes, device_str: str):
    if not blob:
        return
    sd = torch.load(io.BytesIO(blob), map_location=torch.device(device_str))
    agent.q_network.load_state_dict(sd["policy"], strict=True)
    agent.target_network.load_state_dict(sd["policy"], strict=False)


# ====== 워커: 샘플링 + (가능하면) 로컬 학습 ======
def _worker_rollout(args):
    """
    각 워커 태스크는 다음을 수행:
      1) 전달받은 Q-network 파라미터 로드
      2) episodes 개수만큼 에피소드 샘플링
      3) replay buffer 채운 뒤, 가능하면 compute_gradients() 호출
      4) gradients 리스트, 에피소드 리턴/시간 리스트 반환

    시간 측정:
      - sample_times : 각 에피소드별 샘플링 duration (t1_sam - t0_sam) [로그용]
      - compute_times: 각 에피소드별 gradient 계산 duration [로그용]
      - sample_spans : 각 에피소드별 (ep_idx, t0_sam, t1_sam) (Gantt용)
      - compute_spans: 각 에피소드별 (ep_idx, t0_cmp, t1_cmp) (Gantt용)
    """
    wid, start_ep, episodes, max_steps, device_str, batch_size = args
    try:
        # 워커용 DQNAgent
        agent = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=N_ACTIONS,
            batch_size=batch_size,
            min_buffer_size=batch_size
        )
        _load_policy_weights(agent, weights_blob=None, device_str=device_str)  # weights는 아래에서 로드

        # 위 줄 수정: weights_blob는 args에 포함되어야 하므로 다시 언패킹
    except ValueError:
        # 위에서 실수했으니 제대로 다시 정의
        pass


def _worker_rollout(args):
    """
    args:
      wid         : worker id (로그용)
      start_ep    : 이 태스크에서 처리할 첫 global episode index (1-based)
      episodes    : 이 태스크에서 처리할 에피소드 수
      max_steps   : 에피소드당 최대 step 수
      device_str  : 'cpu' / 'cuda'
      batch_size  : DQN batch size
      weights_blob: 전역 Q-network 파라미터 (bytes)
    """
    wid, start_ep, episodes, max_steps, device_str, batch_size, weights_blob = args
    try:
        # 워커용 DQNAgent
        agent = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=N_ACTIONS,
            batch_size=batch_size,
            min_buffer_size=batch_size
        )
        _load_policy_weights(agent, weights_blob, device_str)

        env = JobRoutingGymEnv(Config())

        all_grads      = []   # [grad_dict_ep1, ...]
        ep_rewards     = []   # [return_ep1, ...]
        sample_times   = []   # [duration per episode]
        compute_times  = []   # [duration per episode]
        sample_spans   = []   # [(ep_idx, t0_sam, t1_sam), ...]
        compute_spans  = []   # [(ep_idx, t0_cmp, t1_cmp), ...]

        for ep_offset in range(episodes):
            ep_idx = start_ep + ep_offset  # global episode index

            s, _info = env.reset()
            ep_return = 0.0

            # ----- 샘플링 -----
            t0_sam = time.time()
            for _t in range(max_steps):
                a = agent.select_action(s)
                s2, r, done, trunc, _info = env.step(a)

                agent.replay_buffer.push(s, a, r, s2, float(done))
                ep_return += r
                s = s2

                if done:
                    break
            t1_sam = time.time()
            dur_sam = t1_sam - t0_sam

            sample_times.append(dur_sam)
            sample_spans.append((ep_idx, t0_sam, t1_sam))
            ep_rewards.append(ep_return)

            # ----- 로컬 gradient 계산 (Compute Time) -----
            if len(agent.replay_buffer) >= agent.min_buffer_size:
                t0_cmp = time.time()
                try:
                    grads = agent.compute_gradients(batch_size=agent.batch_size)
                except ValueError:
                    grads = None
                t1_cmp = time.time()

                if grads is not None:
                    dur_cmp = t1_cmp - t0_cmp
                    compute_times.append(dur_cmp)
                    compute_spans.append((ep_idx, t0_cmp, t1_cmp))

                    grads_cpu = {k: v.detach().cpu() for k, v in grads.items()}
                    all_grads.append(grads_cpu)
                else:
                    compute_times.append(0.0)
            else:
                compute_times.append(0.0)

        env.close()
        # return: (wid, grads, rewards, sample_times, compute_times, sample_spans, compute_spans, err)
        return wid, all_grads, ep_rewards, sample_times, compute_times, sample_spans, compute_spans, None

    except Exception:
        # 에러 시 형식 맞춰서 반환
        return wid, None, None, None, None, None, None, traceback.format_exc()


# ====== 워커 로그 ======
def _log_worker_batch(writer, wid, rewards, s_times, c_times, ep_counter):
    avg_r = float(np.mean(rewards)) if rewards else 0.0
    avg_s = float(np.mean(s_times)) if s_times else 0.0
    avg_c = float(np.mean(c_times)) if c_times else 0.0

    print(f"[Worker {wid}] avg_return={avg_r:.3f} | sample(avg)={avg_s:.4f}s | compute(avg)={avg_c:.4f}s")

    if writer is not None:
        writer.add_scalar(f"Worker{wid}/AvgReturn", avg_r, ep_counter)
        writer.add_scalar(f"Worker{wid}/AvgSampleSec", avg_s, ep_counter)
        writer.add_scalar(f"Worker{wid}/AvgComputeSec", avg_c, ep_counter)


# ====== Gantt-style union time 계산 유틸 ======
def _gantt_total_time(intervals):
    """
    intervals: [(start, end), ...] (start, end는 절대 시간 time.time() 값)
    Gantt chart 관점에서 "적어도 하나의 워커가 해당 작업을 수행한 총 시간"
    -> 구간들의 union 길이
    """
    if not intervals:
        return 0.0

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    cur_s, cur_e = intervals[0]

    for s, e in intervals[1:]:
        if s <= cur_e:
            # 겹치면 구간 확장
            cur_e = max(cur_e, e)
        else:
            # 안 겹치면 이전 구간 확정
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    total = sum(e - s for s, e in merged)
    return total


# ====== 메인 (Centralized + Asynchronous) ======
def main(mode: str = "C", workers: int = 5):
    """
    mode:
      - 'A': 샘플링/학습 모두 CUDA (가능한 경우)
      - 'B': 샘플링/학습 모두 CPU
      - 'C': 샘플링 CPU, 학습 CUDA (기본)

    시간 측정 정의:
      - Sampling Time:
          * 각 에피소드마다, 모든 워커의 (sampling 시작, 종료) interval들을 Gantt-style로 union
          * 이렇게 얻은 per-episode sampling 시간을 전 에피소드에 대해 누적합하여 reporting
      - Compute Time:
          * 각 에피소드마다, 모든 워커의 compute_gradients() interval들을 Gantt-style union
          * per-episode compute 시간을 누적합
      - Aggregation Time:
          * global model이 워커로부터 gradient를 받고
            apply_gradients()로 파라미터를 업데이트하는 데 걸린 시간들의 단순 합
      - Total Time:
          * main() 시작부터 종료까지의 전체 벽시계(wall-clock) 시간
    """
    mode = (mode or "C").upper()
    if mode not in ("A", "B", "C"):
        raise ValueError("mode must be one of {'A','B','C'}")

    infer_dev = "cuda" if mode == "A" else ("cpu" if mode == "B" else "cpu")
    train_dev = "cuda" if mode in ("A", "C") and torch.cuda.is_available() else "cpu"

    run_name = f"{mode}_w{int(workers)}_ASYNC_DQN"
    log_dir = TB_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(log_dir))

    # 전역 Parameter Server
    server = DQNAgent(state_dim=STATE_DIM, action_dim=N_ACTIONS)

    # 통계 변수 (state 딕셔너리로 관리)
    t0_total = time.time()
    state = {
        "episodes_done": 0,         # 실제로 끝난 에피소드 수
        "episodes_issued": 0,       # 워커들에게 할당한 에피소드 수
        "in_flight": 0,
        "server": server,
        "writer": writer,
        # 에피소드별 interval 기록: ep_idx -> [(start, end), ...]
        "sampling_by_ep": defaultdict(list),
        "compute_by_ep": defaultdict(list),
        # Aggregation time (global model update) 누적
        "aggregation_time_acc": 0.0,
        "errors": 0,
    }

    pool = Pool(processes=int(workers))

    # ----- 태스크 제출 -----
    def _submit_task(wid: int):
        # 아직 할당되지 않은 에피소드가 남았는지 확인
        if state["episodes_issued"] >= N_EPISODES:
            return False

        # 이번 태스크에서 처리할 에피소드 수
        episodes = min(EPISODES_PER_TASK, N_EPISODES - state["episodes_issued"])
        start_ep = state["episodes_issued"] + 1  # 1-based global episode index

        blob = _serialize_policy(state["server"])
        args = (wid, start_ep, episodes, MAX_STEPS_PER_EP, infer_dev, server.batch_size, blob)

        pool.apply_async(
            _worker_rollout,
            args=(args,),
            callback=_on_result,
            error_callback=_on_error
        )
        state["episodes_issued"] += episodes
        state["in_flight"] += 1
        return True

    # ----- 항상 worker 수만큼 태스크를 유지 -----
    def _maybe_issue_more():
        while (state["in_flight"] < int(workers)) and (state["episodes_issued"] < N_EPISODES):
            wid = state["episodes_issued"] + state["in_flight"]  # 그냥 로그용 id
            if not _submit_task(wid):
                break

    # ----- 결과 콜백 -----
    def _on_result(res):
        # res: wid, grad_list, rewards, s_times, c_times, s_spans, c_spans, err
        wid, grad_list, rewards, s_times, c_times, s_spans, c_spans, err = res
        try:
            if err is not None:
                print(f"[WorkerError] {wid}\n{err}")
                state["errors"] += 1
            else:
                # 이번 태스크에서 실제로 처리된 에피소드 수
                n_eps = len(rewards) if rewards is not None else EPISODES_PER_TASK
                ep_idx_for_log = state["episodes_done"] + n_eps

                # 워커 로그 (평균 duration 기준)
                _log_worker_batch(state["writer"], wid, rewards, s_times, c_times, ep_counter=ep_idx_for_log)

                # ----- 에피소드별 Gantt interval 누적 -----
                if s_spans:
                    for ep_idx, t0, t1 in s_spans:
                        state["sampling_by_ep"][ep_idx].append((t0, t1))
                if c_spans:
                    for ep_idx, t0, t1 in c_spans:
                        state["compute_by_ep"][ep_idx].append((t0, t1))

                # ----- global model aggregation (apply_gradients) -----
                # gradient 평균 없이, 들어온 것들을 즉시/순차적으로 적용
                if grad_list and len(grad_list) > 0:
                    for grads in grad_list:
                        t0_apply = time.time()
                        state["server"].apply_gradients(grads)
                        t1_apply = time.time()
                        state["aggregation_time_acc"] += (t1_apply - t0_apply)

                # 에피소드 수 증가
                state["episodes_done"] += n_eps

        finally:
            state["in_flight"] -= 1
            _maybe_issue_more()

    # ----- 에러 콜백 -----
    def _on_error(exc):
        print("[Async Error]", exc)
        state["errors"] += 1
        state["in_flight"] -= 1
        _maybe_issue_more()

    # 초기 태스크 발행
    _maybe_issue_more()

    # ===== 메인 루프 =====
    try:
        last_ckpt_step = 0
        while (state["episodes_done"] < N_EPISODES) or (state["in_flight"] > 0):
            time.sleep(0.01)

            # 체크포인트 저장
            if SAVE_EVERY and state["episodes_done"] >= SAVE_EVERY:
                step = state["episodes_done"] // SAVE_EVERY
                if step > last_ckpt_step:
                    last_ckpt_step = step
                    torch.save({"policy": state["server"].q_network.state_dict()}, str(CKPT_PTH))
                    print(f"[CKPT] saved at {state['episodes_done']} episodes")

        # 최종 저장
        torch.save({"policy": state["server"].q_network.state_dict()}, str(CKPT_PTH))

        t1_total = time.time()
        total_time = t1_total - t0_total

        # ===== per-episode Gantt union 후 누적합 계산 =====
        sampling_total = 0.0
        compute_total  = 0.0

        # 에피소드 index는 1 ~ N_EPISODES
        for ep in range(1, N_EPISODES + 1):
            sam_intervals = state["sampling_by_ep"].get(ep, [])
            cmp_intervals = state["compute_by_ep"].get(ep, [])

            sampling_total += _gantt_total_time(sam_intervals)
            compute_total  += _gantt_total_time(cmp_intervals)

        aggregation_total = state["aggregation_time_acc"]

        print("\n===== SUMMARY (DQN Centralized ASYNCHRONOUS) =====")
        print(f"Episodes        : {state['episodes_done']}")
        print(f"Sampling Time   : {sampling_total:.3f}s  (per-episode Gantt union 누적합)")
        print(f"Compute Time    : {compute_total:.3f}s  (per-episode Gantt union 누적합)")
        print(f"Aggregation Time: {aggregation_total:.3f}s  (Global model apply_gradients 누적)")
        print(f"Errors          : {state['errors']}")
        print(f"Total Time      : {total_time:.3f}s  (wall-clock)")
        print(f"Finished at     : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1_total))}")
        print("===================================================\n")

        return {
            "mode": mode,
            "workers": int(workers),
            "episodes": int(state["episodes_done"]),
            "sampling_time_s": float(sampling_total),
            "compute_time_s": float(compute_total),
            "aggregation_time_s": float(aggregation_total),
            "total_time_s": float(total_time),
            "errors": int(state["errors"]),
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
        ("C", 1),
        ("C", 5),
        ("A", 1),
        ("A", 5),
        ("B", 1),
        ("B", 5),
    ]

    out_csv = Path("dqn_run_summary_async.csv")
    fieldnames = [
        "mode", "workers", "episodes",
        "sampling_time_s", "compute_time_s", "aggregation_time_s",
        "total_time_s", "errors",
        "finished_at_ts", "finished_at", "tb_log_dir"
    ]

    results = []
    for m, w in run_plan:
        print(f"\n### RUN start: mode={m}, workers={w} (DQN ASYNC) ###\n")
        res = main(m, workers=w)
        if res is None:
            res = {
                "mode": m, "workers": w, "episodes": 0,
                "sampling_time_s": 0.0, "compute_time_s": 0.0, "aggregation_time_s": 0.0,
                "total_time_s": 0.0, "errors": 1,
                "finished_at_ts": time.time(),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "tb_log_dir": str(TB_DIR / f"{m}_w{int(w)}_ASYNC_DQN"),
            }
        results.append(res)

        write_header = not out_csv.exists()
        with out_csv.open("a", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                csv_writer.writeheader()
            csv_writer.writerow({k: res.get(k) for k in fieldnames})

    print(f"\n[CSV] saved/updated → {out_csv}\n")
