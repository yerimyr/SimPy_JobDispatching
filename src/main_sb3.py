from __future__ import annotations
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from environment import JobRoutingGymEnv, Config
from log import EpisodeLogger

N_EPISODES = 1000

def make_env(SEED=0):
    env = JobRoutingGymEnv(Config(SEED=SEED))
    env = Monitor(env)  
    return env


def evaluate(model: DQN, env: gym.Env, n_episodes=10):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ret += reward
        returns.append(ret)
    return float(np.mean(returns)), float(np.std(returns))

def get_new_logdir(base_dir="runs/tb", prefix="sb3_JobRouting"):
    os.makedirs(base_dir, exist_ok=True)
    i = 1
    while True:
        logdir = os.path.join(base_dir, f"{prefix}{i}")
        if not os.path.exists(logdir):
            return logdir
        i += 1


class TensorboardEpisodeCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_count = 0
        logdir = get_new_logdir()
        print(f"[TensorBoard] logging to {logdir}")
        self.writer = SummaryWriter(logdir)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info.keys():
                self.ep_count += 1
                ep_rew = info["episode"]["r"]
                ep_len = info["episode"]["l"]
                avg_reward = ep_rew / ep_len if ep_len > 0 else ep_rew

                print(f"[Episode {self.ep_count}] reward={avg_reward:.2f}")

                self.writer.add_scalar("train/avg_reward", avg_reward, self.ep_count)

                last_loss = self.model.logger.name_to_value.get("train/loss")
                if last_loss is not None:
                    self.writer.add_scalar("train/loss", last_loss, self.ep_count)
        return True

    def _on_training_end(self) -> None:
        self.writer.close()


def main():
    cfg = Config(SEED=0)
    NUM_JOBS = cfg.NUM_JOBS
    total_timesteps = NUM_JOBS * N_EPISODES

    os.makedirs("runs", exist_ok=True)
    logger = EpisodeLogger("runs/episode_log.csv")

    env = make_env(SEED=0)
    eval_env = make_env(SEED=123)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=5000,
        learning_starts=0,
        batch_size=64,
        gamma=1.0,
        train_freq=1,
        target_update_interval=500,
        verbose=0,  
        tensorboard_log=None  
    )

    TIMESTEPS = total_timesteps
    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=False,
        callback=TensorboardEpisodeCallback()
    )

    mean_ret, std_ret = evaluate(model, eval_env, n_episodes=20)
    print(f"[EVAL] Return mean={int(mean_ret):,} ± {int(std_ret):,}")

    model.save("runs/dqn_job_routing_model")


if __name__ == "__main__":
    main()
