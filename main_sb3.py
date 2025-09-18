# main_train.py
# Stable-Baselines3의 DQN으로 학습 예시
# 설치 필요: pip install simpy gymnasium stable-baselines3[extra]

from __future__ import annotations
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from environment import JobRoutingGymEnv
from environment import Config
from log import EpisodeLogger


def make_env(seed=0):  
    """학습/평가에 사용할 환경 인스턴스를 생성"""
    env = JobRoutingGymEnv(Config(seed=seed))
    env = Monitor(env)  # episode rewards, length 자동 기록
    return env


def evaluate(model: DQN, env: gym.Env, n_episodes=10):
    """학습된 model을 주어진 env에서 n회 평가"""
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ret += reward
            length += 1
        returns.append(ret)

    return float(np.mean(returns)), float(np.std(returns))


def main():
    # 로그 디렉토리 생성
    os.makedirs("runs", exist_ok=True)
    logger = EpisodeLogger("runs/episode_log.csv")

    # 환경 생성
    env = make_env(seed=0)
    eval_env = make_env(seed=123)

    # DQN 모델 초기화
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=5000,
        learning_starts=100,
        batch_size=64,
        gamma=1.0,            # 문제 정의에 맞춰 감가 없음
        train_freq=1,
        target_update_interval=500,
        verbose=1,
        tensorboard_log="runs/tb"
    )

    # 학습 (TensorBoard 로그 남김)
    TIMESTEPS = 20000
    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=False,
        tb_log_name="sb3_JobRouting"  # TensorBoard에서 확인 가능
    )

    # 평가
    mean_ret, std_ret = evaluate(model, eval_env, n_episodes=20)
    print(f"[EVAL] Return mean={int(mean_ret):,} ± {int(std_ret):,} (<= 0가 이상적)")

    # 에피소드별 로그 기록 (데모)
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        L = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ret += reward
            L += 1
        logger.log(ep, ret, L)
        print(f"Episode {ep}: return={int(ret):,}, length={L}")

    # 모델 저장
    model.save("runs/dqn_job_routing_model")


if __name__ == "__main__":
    main()
