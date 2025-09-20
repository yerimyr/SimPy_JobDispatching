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
    INTERARRIVAL_SEC = 60
    PROC_TIME_SEC = 180
    SEED = 0
    
    LEARNING_RATE = 1e-3
    GAMMA = 0.99
    CLIP_EPSILON = 0.2
    UPDATE_STEPS = 10
    GAE_LAMBDA = 0.95
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    BATCH_SIZE = 20
    
    #DEVICE = "cpu"
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

        self.jobs_generated = 0  # 지금까지 도착한 총 job의 개수
        self.jobs_completed = 0  # 지금까지 완료된 총 job의 개수
        self.last_obs = None  # 상태 전이 관리
        self.pending_job = None  # 아직 server에 배정되지 않고 대기 중인 job을 담아둠

        self.env.process(self._job_arrival_proc())
        self.server_finish_times = [0.0 for _ in range(self.cfg.NUM_SERVERS)]
        
        self.on_job_finish = None


    def _job_arrival_proc(self):
        for i in range(self.cfg.NUM_JOBS):
            if i > 0:
                yield self.env.timeout(self.cfg.INTERARRIVAL_SEC)
            self.jobs_generated += 1
            self.pending_job = {"id": i, "arrival_time": self.env.now}

            while self.pending_job is not None:
                yield self.env.timeout(0)  # 다른 프로세스에게 양보


    def _job_process(self, server_id: int, job_id: int):
        with self.servers[server_id].request() as req:  # server에 요청 이벤트를 전달
            yield req  # 이 요청 req이 완료되어 server 리소스를 점유할 때까지 기다림
            finish_time = self.env.now + self.cfg.PROC_TIME_SEC
            self.server_finish_times[server_id] = finish_time  # 예상 종료 시각을 서버 별 상태 배열에 기록
            yield self.env.timeout(self.cfg.PROC_TIME_SEC)  # 처리 시간만큼 진행
            self.jobs_completed += 1
            self.server_finish_times[server_id] = 0.0  # 해당 서버가 대기 상태가 되었음을 표시

            if self.on_job_finish is not None:
                self.on_job_finish(job_id, server_id, self.env.now)

    def _observe(self):
        r, q = [], []  # 각 서버의 남은 처리 시간, 각 서버의 대기열 길이를 담을 리스트
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
        r = obs[:num_servers]               # 남은 처리시간 벡터
        q = obs[num_servers:2*num_servers]  # 큐 길이 벡터

        pk = np.array([
            r[i] + (q[i] + 1) * self.cfg.PROC_TIME_SEC
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

        r_high = [self.core.cfg.PROC_TIME_SEC] * num_servers

        q_high = [self.core.cfg.NUM_JOBS] * num_servers

        completed_high = [self.core.cfg.NUM_JOBS]

        time_high = [
            self.core.cfg.NUM_JOBS * self.core.cfg.INTERARRIVAL_SEC
            + self.core.cfg.NUM_JOBS * self.core.cfg.PROC_TIME_SEC
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
        print(f"t={self.core.env.now:.4f}, obs={self.core.last_obs}, completed={self.core.jobs_completed}")
