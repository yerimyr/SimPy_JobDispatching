from __future__ import annotations
import numpy as np
import simpy
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import torch

@dataclass
class Config:
    NUM_SERVERS = 3
    NUM_JOBS = 20
    INTERARRIVAL_HOUR = 60 / 3600     # 60초 = 1분 = 1/60시간
    PROC_TIME_HOUR = 180 / 3600       # 180초 = 3분 = 0.05시간
    SEED = 0
    LEARNING_RATE = 0.005 #0.0020114611312585335
    GAMMA = 0.9217097681003058
    CLIP_EPSILON = 0.058307315213304435
    UPDATE_STEPS = 6
    GAE_LAMBDA = 0.8903849352860319 
    ENT_COEF = 0.010086246747868191
    VF_COEF = 0.2863692146258702
    MAX_GRAD_NORM = 0.5
    BATCH_SIZE = 20
    HIDDEN_SIZE = 16
    STATE_DIM = 8
    N_ACTIONS = 3

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class JobRoutingSimPyEnv:
    KEYS = [
        "S1_Remain", "S2_Remain", "S3_Remain",
        "S1_Queue", "S2_Queue", "S3_Queue",
        "Completed", "TimeNow"
    ]

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.rng = np.random.default_rng(self.cfg.SEED)
        self.env = simpy.Environment()

        self.servers = [simpy.Resource(self.env, capacity=1)
                        for _ in range(self.cfg.NUM_SERVERS)]

        self.jobs_generated = 0  
        self.jobs_completed = 0  
        self.last_obs = None  
        self.pending_job = None  

        self.env.process(self._job_arrival_proc())
        self.server_finish_times = [0.0 for _ in range(self.cfg.NUM_SERVERS)]
        self.on_job_finish = None

    def _job_arrival_proc(self):
        for i in range(self.cfg.NUM_JOBS):
            if i > 0:
                yield self.env.timeout(self.cfg.INTERARRIVAL_HOUR)
            self.jobs_generated += 1
            self.pending_job = {"id": i, "arrival_time": self.env.now}
            while self.pending_job is not None:
                yield self.env.timeout(0)

    def _job_process(self, server_id: int, job_id: int):
        with self.servers[server_id].request() as req:
            yield req
            finish_time = self.env.now + self.cfg.PROC_TIME_HOUR
            self.server_finish_times[server_id] = finish_time
            yield self.env.timeout(self.cfg.PROC_TIME_HOUR)
            self.jobs_completed += 1
            self.server_finish_times[server_id] = 0.0

            if self.on_job_finish is not None:
                self.on_job_finish(job_id, server_id, self.env.now)

    def _observe(self):
        r, q = [], []
        for k, s in enumerate(self.servers):
            remain = max(0.0, self.server_finish_times[k] - self.env.now)
            qk = len(s.queue)
            r.append(remain)
            q.append(qk)
        obs = np.array(
            r + q + [float(self.jobs_completed), float(self.env.now)],  
            dtype=np.float32
        )
        self.last_obs = obs
        return obs

    def _reward(self, obs, action):
        num_servers = self.cfg.NUM_SERVERS
        r = obs[:num_servers]
        q = obs[num_servers:2*num_servers]
        pk = np.array([
            r[i] + (q[i] + 1) * self.cfg.PROC_TIME_HOUR
            for i in range(num_servers)
        ])
        return float(pk.min() - pk[int(action)])

    def reset(self, SEED: int | None = None):
        if SEED is not None:
            self.rng = np.random.default_rng(SEED)
        self.env = simpy.Environment()
        self.servers = [simpy.Resource(self.env, capacity=1)
                        for _ in range(self.cfg.NUM_SERVERS)]
        self.jobs_generated = 0
        self.jobs_completed = 0
        self.pending_job = None
        self.last_obs = None
        self.env.process(self._job_arrival_proc())
        return self._observe()

    def step(self, action: int):
        while self.pending_job is None and self.jobs_generated < self.cfg.NUM_JOBS:
            self.env.step()

        if self.pending_job is None and self.jobs_generated >= self.cfg.NUM_JOBS:
            while any(len(s.users) > 0 or len(s.queue) > 0 for s in self.servers):
                self.env.step()
            obs = self._observe()
            reward = 0.0
            done = True
            info = {"completed": self.jobs_completed}
            return obs, reward, done, info

        obs_before = self._observe()
        job = self.pending_job
        self.pending_job = None
        self.env.process(self._job_process(action, job["id"]))
        reward = self._reward(obs_before, action)

        while (
            self.pending_job is None
            and not (
                self.jobs_completed == self.cfg.NUM_JOBS
                and all(len(s.users) == 0 for s in self.servers)
            )
        ):
            self.env.step()

        obs = self._observe()
        done = (
            self.jobs_completed == self.cfg.NUM_JOBS
            and all(len(s.users) == 0 and len(s.queue) == 0 for s in self.servers)
        )
        info = {"completed": self.jobs_completed}
        return obs, reward, done, info


class JobRoutingGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Config | None = None):
        super().__init__()
        self.core = JobRoutingSimPyEnv(config)
        num_servers = self.core.cfg.NUM_SERVERS

        r_high = [self.core.cfg.PROC_TIME_HOUR] * num_servers
        q_high = [self.core.cfg.NUM_JOBS] * num_servers
        completed_high = [self.core.cfg.NUM_JOBS]
        time_high = [
            self.core.cfg.NUM_JOBS * self.core.cfg.INTERARRIVAL_HOUR
            + self.core.cfg.NUM_JOBS * self.core.cfg.PROC_TIME_HOUR
        ]

        high = np.array(r_high + q_high + completed_high + time_high, dtype=np.float32)
        low = np.zeros_like(high, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(self.core.cfg.NUM_SERVERS)

    def reset(self, *, seed: int | None = None, options=None):
        obs = self.core.reset(SEED=seed if seed is not None else None)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.core.step(action)
        return obs, reward, done, False, info

    def render(self):
        print(f"t={self.core.env.now:.2f} h, obs={self.core.last_obs}, completed={self.core.jobs_completed}")
