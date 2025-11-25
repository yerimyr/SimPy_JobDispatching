# main_DQN_centralized_asynchronous.py
import os, io, time, csv, traceback
from pathlib import Path
from multiprocessing import Pool, get_start_method
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
SAVE_EVERY        = 200        # 너무 자주 저장되던 거 간격 조금 늘림


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
      2) EPISODES_PER_TASK 개수만큼 에피소드 샘플링
      3) replay buffer 채운 뒤, 가능하면 compute_gradients() 호출
      4) gradients 리스트, 에피소드 리턴/시간 리스트 반환
    """
    wid, weights_blob, episodes, max_steps, device_str, batch_size = args
    try:
        # 워커용 DQNAgent (min_buffer_size를 batch_size 수준으로 낮춰서 학습이 실제로 돌도록)
        agent = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=N_ACTIONS,
            batch_size=batch_size,
            min_buffer_size=batch_size  # 한 에피소드에 batch_size 이상 transition 쌓이면 곧바로 학습 가능
        )
        _load_policy_weights(agent, weights_blob, device_str)

        env = JobRoutingGymEnv(Config())

        all_grads   = []   # [grad_dict_ep1, ...]
        ep_rewards  = []   # [return_ep1, ...]
        sample_times = []  # [샘플링 시간(ep 단위), ...]
        learn_times  = []  # [로컬 학습 시간(ep 단위), ...]

        for _ in range(episodes):
            s, _info = env.reset()
            ep_return = 0.0

            # ----- 샘플링 -----
            t0_sam = time.time()
            for _t in range(max_steps):
                a = agent.select_action(s)
                s2, r, done, trunc, _info = env.step(a)

                # replay buffer에 transition 저장
                agent.replay_buffer.push(s, a, r, s2, float(done))
                ep_return += r
                s = s2

                if done:
                    break
            t1_sam = time.time()
            sample_times.append(t1_sam - t0_sam)
            ep_rewards.append(ep_return)

            # ----- 로컬 gradient 계산 -----
            if len(agent.replay_buffer) >= agent.min_buffer_size:
                t0_learn = time.time()
                try:
                    grads = agent.compute_gradients(batch_size=agent.batch_size)
                except ValueError:
                    # replay buffer가 여전히 부족하면 스킵
                    grads = None
                t1_learn = time.time()
                learn_times.append(t1_learn - t0_learn)

                if grads is not None:
                    grads_cpu = {k: v.detach().cpu() for k, v in grads.items()}
                    all_grads.append(grads_cpu)
            else:
                # 아직 버퍼가 작아 학습 못하는 경우라도 시간/리턴은 기록
                learn_times.append(0.0)

        env.close()
        return wid, all_grads, ep_rewards, sample_times, learn_times, None

    except Exception:
        return wid, None, None, None, None, traceback.format_exc()


# ====== 평균 그라디언트 ======
def _average_gradients(grad_lists):
    if len(grad_lists) == 0:
        return {}

    keys = set()
    for gd in grad_lists:
        keys.update(gd.keys())

    avg = {}
    for k in keys:
        terms = [gd[k] for gd in grad_lists if k in gd]
        if len(terms) == 0:
            continue
        stacked = torch.stack(terms, dim=0)
        avg[k] = stacked.mean(dim=0)
    return avg


# ====== 워커 로그 (PPO와 비슷한 포맷) ======
def _log_worker_batch(writer, wid, rewards, s_times, l_times, ep_counter):
    avg_r = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
    avg_s = float(np.mean(s_times)) if len(s_times) > 0 else 0.0
    avg_l = float(np.mean(l_times)) if len(l_times) > 0 else 0.0

    print(f"[Worker {wid}] avg_return={avg_r:.3f} | sample={avg_s:.3f}s | learn={avg_l:.3f}s")

    if writer is not None:
        writer.add_scalar(f"Worker{wid}/AvgReturn", avg_r, ep_counter)
        writer.add_scalar(f"Worker{wid}/AvgSampleSec", avg_s, ep_counter)
        writer.add_scalar(f"Worker{wid}/AvgLearnSec", avg_l, ep_counter)


# ====== 메인 (Centralized + Asynchronous) ======
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
        "episodes_done": 0,
        "in_flight": 0,
        "server": server,
        "writer": writer,
        "sampling_time_acc": 0.0,
        "aggregation_time_acc": 0.0,
        "apply_time_acc": 0.0,
        "errors": 0,
    }

    pool = Pool(processes=int(workers))

    # ----- 태스크 제출 -----
    def _submit_task(wid: int):
        if state["episodes_done"] >= N_EPISODES:
            return False

        blob = _serialize_policy(state["server"])
        args = (wid, blob, EPISODES_PER_TASK, MAX_STEPS_PER_EP, infer_dev, server.batch_size)

        pool.apply_async(
            _worker_rollout,
            args=(args,),
            callback=_on_result,
            error_callback=_on_error
        )
        state["in_flight"] += 1
        return True

    # ----- 결과 콜백 -----
    def _on_result(res):
        wid, grad_list, rewards, s_times, l_times, err = res
        try:
            if err is not None:
                print(f"[WorkerError] {wid}\n{err}")
                state["errors"] += 1
            else:
                # 이번 태스크에서 실제로 처리된 에피소드 수
                n_eps = len(rewards) if rewards is not None else EPISODES_PER_TASK
                ep_idx = state["episodes_done"] + n_eps

                # 워커 로그
                _log_worker_batch(state["writer"], wid, rewards, s_times, l_times, ep_counter=ep_idx)

                # 샘플링 시간 누적 (에피소드 평균 기준)
                if s_times:
                    state["sampling_time_acc"] += float(np.mean(s_times))

                # gradient가 있으면 평균 내고 바로 전역 모델 업데이트
                if grad_list and len(grad_list) > 0:
                    t0_agg = time.time()
                    avg_grad = _average_gradients(grad_list)
                    t1_agg = time.time()
                    state["aggregation_time_acc"] += (t1_agg - t0_agg)

                    t0_apply = time.time()
                    state["server"].apply_gradients(avg_grad)
                    t1_apply = time.time()
                    state["apply_time_acc"] += (t1_apply - t0_apply)

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

    # ----- 항상 worker 수만큼 태스크를 유지 -----
    def _maybe_issue_more():
        while (state["in_flight"] < int(workers)) and (state["episodes_done"] < N_EPISODES):
            wid = state["episodes_done"] + state["in_flight"]  # 그냥 로그용 id
            if not _submit_task(wid):
                break

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
                    print(f"[ckpt] saved → {CKPT_PTH} (episodes_done={state['episodes_done']})")

        # 최종 저장
        torch.save({"policy": state["server"].q_network.state_dict()}, str(CKPT_PTH))

        t1_total = time.time()
        total_time = t1_total - t0_total

        print("\n===== SUMMARY (DQN Centralized ASYNCHRONOUS) =====")
        print(f"Episodes        : {state['episodes_done']}")
        print(f"Sampling Time   : {state['sampling_time_acc']:.3f}s")
        print(f"Aggregation Time: {state['aggregation_time_acc']:.3f}s")
        print(f"Apply Step Time : {state['apply_time_acc']:.3f}s")
        print(f"Errors          : {state['errors']}")
        print(f"Total Time      : {total_time:.3f}s")
        print(f"Finished at     : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1_total))}")
        print("===================================================\n")

        return {
            "mode": mode,
            "workers": int(workers),
            "episodes": int(state["episodes_done"]),
            "sampling_time_s": float(state["sampling_time_acc"]),
            "aggregation_time_s": float(state["aggregation_time_acc"]),
            "apply_time_s": float(state["apply_time_acc"]),
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
        "sampling_time_s", "aggregation_time_s", "apply_time_s",
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
                "sampling_time_s": 0.0, "aggregation_time_s": 0.0, "apply_time_s": 0.0,
                "total_time_s": 0.0, "errors": 1,
                "finished_at_ts": time.time(),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "tb_log_dir": str(TB_DIR / f"{m}_w{int(w)}_ASYNC_DQN"),
            }
        results.append(res)

        write_header = not out_csv.exists()
        with out_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: res.get(k) for k in fieldnames})

    print(f"\n[CSV] saved/updated → {out_csv}\n")
