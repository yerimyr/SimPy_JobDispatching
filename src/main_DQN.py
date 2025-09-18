# main_train.py
# Custom DQNAgent 학습 예시 (SB3 대신 직접 구현한 버전)
# 설치 필요: pip install simpy gymnasium torch tensorboard

import os
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from environment import JobRoutingGymEnv, Config
from DQN import DQNAgent, BATCH_SIZE, TOTAL_TIMESTEPS


def evaluate(agent: DQNAgent, env: gym.Env, n_episodes=10):
    """학습된 에이전트를 평가"""
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
    """실행할 때마다 DQN_JobRouting1,2,... 폴더 생성"""
    os.makedirs(base_dir, exist_ok=True)
    i = 1
    while True:
        logdir = os.path.join(base_dir, f"{prefix}{i}")
        if not os.path.exists(logdir):
            return logdir
        i += 1


def main():
    # 로그 디렉토리
    logdir = get_new_logdir()
    writer = SummaryWriter(logdir)
    print(f"[TensorBoard] logging to {logdir}")

    # 환경 생성
    env = JobRoutingGymEnv(Config(seed=0))
    eval_env = JobRoutingGymEnv(Config(seed=123))

    # 상태/행동 크기
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # DQN 에이전트
    agent = DQNAgent(state_dim=state_dim, action_dim=n_actions)

    # 학습
    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    ep_count = 0

    for t in range(1, TOTAL_TIMESTEPS + 1):
        # 행동 선택
        action = agent.select_action(obs)

        # 환경 스텝
        obs2, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # transition 저장
        agent.replay_buffer.push(obs, action, reward, obs2, done)

        # 학습
        loss = agent.update(batch_size=BATCH_SIZE)

        obs = obs2
        ep_ret += reward
        ep_len += 1

        # TensorBoard 기록
        if loss is not None:
            writer.add_scalar("train/loss", loss, t)
        writer.add_scalar("rollout/reward", reward, t)

        # episode 끝나면 기록
        if done:
            ep_count += 1
            writer.add_scalar("rollout/ep_return", ep_ret, ep_count)
            writer.add_scalar("rollout/ep_length", ep_len, ep_count)
            print(f"[Episode {ep_count}] return={int(ep_ret):,}, length={ep_len}")

            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        # SB3처럼 일정 주기로 로그 출력
        if t % 1000 == 0:
            print(f"[{t}/{TOTAL_TIMESTEPS}] steps finished")

    # 평가
    mean_ret, std_ret = evaluate(agent, eval_env, n_episodes=20)
    print(f"[EVAL] Return mean={int(mean_ret):,} ± {int(std_ret):,} (<= 0가 이상적)")

    # 모델 저장
    agent.save("runs/dqn_agent.pt")
    writer.close()


if __name__ == "__main__":
    main()
