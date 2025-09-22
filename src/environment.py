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
    LEARNING_RATE = 0.0003
    GAMMA = 0.99
    CLIP_EPSILON = 0.03
    UPDATE_STEPS = 8
    GAE_LAMBDA = 0.95
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    BATCH_SIZE = 20
    HIDDEN_SIZE = 32
    
    #DEVICE = "cpu"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class JobRoutingSimPyEnv:
    """
    Simulation environment for job routing using SimPy.
    It models multiple servers, job arrivals, and job processing dynamics.

    Attributes:
        cfg (Config): Configuration object with simulation parameters.
        rng (np.random.Generator): Random number generator initialized with the given seed.
        env (simpy.Environment): SimPy environment to simulate time progression.
        servers (list[simpy.Resource]): List of servers handling job processing.
        jobs_generated (int): Number of jobs that have arrived so far.
        jobs_completed (int): Number of jobs that have been completed so far.
        last_obs (np.ndarray): Last observed state vector.
        pending_job (dict | None): Job waiting to be assigned to a server.
        server_finish_times (list[float]): Finish time for each server.
        on_job_finish (callable | None): Callback function triggered when a job finishes.
    """

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
        """
        Job arrival process that generates jobs at fixed interarrival times.
        New jobs are stored in `pending_job` until assigned to a server.
        """
        for i in range(self.cfg.NUM_JOBS):
            if i > 0:
                yield self.env.timeout(self.cfg.INTERARRIVAL_SEC)
            self.jobs_generated += 1
            self.pending_job = {"id": i, "arrival_time": self.env.now}

            while self.pending_job is not None:
                yield self.env.timeout(0)  

    def _job_process(self, server_id: int, job_id: int):
        """
        Simulate processing of a job by a specific server.

        Args:
            server_id (int): ID of the server that processes the job.
            job_id (int): ID of the job being processed.
        """
        with self.servers[server_id].request() as req:  
            yield req  
            finish_time = self.env.now + self.cfg.PROC_TIME_SEC
            self.server_finish_times[server_id] = finish_time  
            yield self.env.timeout(self.cfg.PROC_TIME_SEC)  
            self.jobs_completed += 1
            self.server_finish_times[server_id] = 0.0  

            if self.on_job_finish is not None:
                self.on_job_finish(job_id, server_id, self.env.now)

    def _observe(self):
        """
        Construct the current observation of the environment state.

        Returns:
            np.ndarray: State vector including server remaining times,
                        queue lengths, completed jobs, and current time.
        """
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
        """
        Compute the reward for assigning the pending job to a server.

        Args:
            obs (np.ndarray): Current observation of the system.
            action (int): Selected server ID.

        Returns:
            float: Reward value.
        """
        num_servers = self.cfg.NUM_SERVERS
        r = obs[:num_servers]               
        q = obs[num_servers:2*num_servers]  
        pk = np.array([
            r[i] + (q[i] + 1) * self.cfg.PROC_TIME_SEC
            for i in range(num_servers)
        ])

        return float(pk.min() - pk[int(action)])

    def reset(self, SEED: int | None = None):
        """
        Reset the environment to its initial state.

        Args:
            SEED (int | None): Optional random seed for reproducibility.

        Returns:
            np.ndarray: Initial observation after reset.
        """
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
        """
        Take one step in the environment by assigning the pending job
        to the specified server.

        Args:
            action (int): ID of the server to assign the job to.

        Returns:
            tuple:
                - obs (np.ndarray): Next observation.
                - reward (float): Reward for the chosen action.
                - done (bool): Whether the episode has finished.
                - info (dict): Additional information (e.g., number of completed jobs).
        """
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
    """
    OpenAI GymWrapper for the Job Routing environment.
    This class provides a Gym API on top of the SimPy-based simulation.

    Attributes:
        core (JobRoutingSimPyEnv): Core simulation environment.
        observation_space (gym.spaces.Box): Continuous observation space for states.
        action_space (gym.spaces.Discrete): Discrete action space (server selection).
    """
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