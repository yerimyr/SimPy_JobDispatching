from __future__ import annotations
import numpy as np
import simpy
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

@dataclass
class Config:
    NUM_SERVERS: int = 3
    NUM_JOBS: int = 20
    INTERARRIVAL_SEC: float = 60
    PROC_TIME_SEC: float = 180
    TIME_UNIT: str = "second"
    SEED: int = 0

class JobRoutingSimPyEnv:
    KEYS = ["S1_Remain","S2_Remain","S3_Remain","S1_Queue","S2_Queue","S3_Queue","Completed","TimeNow"]
    def __init__(self, config: Config|None=None):
        self.cfg = config or Config()
        self.env = simpy.Environment()
        self.rng = np.random.default_rng(self.cfg.SEED)
        self.t = 0.0
        self.completed = 0
        self.W = np.zeros(self.cfg.NUM_SERVERS, dtype=np.float64)
        self.arrival_times = None
        self.arr_idx = 0
        self.pending_job = False
        self.last_obs = None

    def _rk_qk_from_W(self, Wk: float):
        """
        wk: 서버 k의 총 작업량 (Workload)
        rk: 서버 k의 현재 실행 중인 job의 잔여 처리시간 (Remaining)
        qk: 서버 k의 대기열(job queue) 길이
        pk: 서버 k의 예상 완료 시간
        """
        if Wk <= 1e-12:  
            return 0.0, 0  # 해당 서버 k의 총 남은 작업량 시간(wk)가 거의 0이면, 남은 처리 시간(rk)=0.0, 대기열 길이(qk)=0으로 반환
        jobs = int(np.ceil(Wk / self.cfg.PROC_TIME_SEC))  # 현재 서버 k에 남아 있는 작업 개수를 계산, 각 job의 서비스 시간이 PROC_TIME_SEC으로 동일하므로, ceil(wk / PROC_TIME_SEC)으로 계산
        rk = Wk - (jobs-1)*self.cfg.PROC_TIME_SEC  # 현재 실행 중인 job의 남은 처리시간 계산
        if rk <= 1e-12:  
            rk = self.cfg.PROC_TIME_SEC  # 만약 현재 실행 중인 job의 남은 처리시간이 거의 0이 되면 다시 rk를 PROC_TIME_SEC으로 되돌림
        qk = max(0, jobs-1)  # 대기열 길이 qk는 전체 job 수에서 현재 실행 중인 1개를 뺸 값이며, 음수를 방지하기 위하여 max(0, jobs-1)로 계산
        return rk, qk  # rk, qk 반환

    def _observe(self):
        r, q = [], []
        for k in range(self.cfg.NUM_SERVERS):
            rk, qk = self._rk_qk_from_W(self.W[k])  
            r.append(rk); q.append(qk)  # 모든 서버 k에 대하여 방금 위 함수로 rk, qk 복원하여 리스트에 담음
        obs = np.array([r[0],r[1],r[2], q[0],q[1],q[2], float(self.completed), float(self.t)], dtype=np.float32)  # 관측 벡터 구성: r1, r2, r3, q1, q2, q3, 완료 수 c, 현재 시간 t
        self.last_obs = obs.copy()
        return obs  # 최근 관측을 저장하고 반환

    def _pk(self, rk, qk):  # 지금 들어가는 job이 해당 서버에서 완료되기 까지의 예상 시간인 pk 계산
        return rk + (qk + 1) * self.cfg.PROC_TIME_SEC  # 현재 실행 중인 job의 남은 시간인 rk + 큐에 있는 qk개와 내 job 1개까지 총 qk+1개의 정해진 서비스 시간을 더함

    def _reward(self, obs, action):
        r = obs[:3]; q = obs[3:6]  # 관측 벡터에서 각 서버의 rk, qk를 분리
        pk = np.array([ self._pk(r[i], q[i]) for i in range(3) ])  # 각 서버의 예상 완료 시간인 pk 벡터 계산
        return float(pk.min() - pk[int(action)])  # 보상은 가장 좋은 서버를 고르면 0, 더 나쁜 서버를 고르면 음수(패널티)

    def _advance_until(self, t_next):
        delta = t_next - self.t
        if delta <= 0:
            self.t = t_next
            return  # 현재 시간 self.t에서 t_next까지 시간을 전진할 때, delta가 0 이하이면 그냥 시간만 맞추고 종료(아무 일도 안 일어남)
        proc = self.cfg.PROC_TIME_SEC
        for k in range(self.cfg.NUM_SERVERS):
            before = self.W[k]
            after = max(0.0, before - delta)  # 각 서버의 총 남은 작업량 w[k]를 delta만큼 감소시킴
            jobs_before = int(np.ceil(before / proc)) if before > 0 else 0
            jobs_after  = int(np.ceil(after  / proc)) if after  > 0 else 0
            self.completed += max(0, jobs_before - jobs_after)  # 전진 전/후의 남은 job 개수를 각각 구해서, 그 차이만큼이 이 구간에서 완료된 job 수
            self.W[k] = after  # 서버 k의 총 남은 작업량 갱신
        self.t = t_next  # 전역 시간 t를 t_next로 이동

    def reset(self, SEED=None):  # 에피소드를 새로 시작하는 함수, 선택적으로 난수 시드를 받을 수 있음
        if SEED is not None:
            self.rng = np.random.default_rng(SEED)  # 시드가 주어지면 난수 발생기(self.rng)를 해당 시드로 재생성 -> 재현성 보장
        self.env = simpy.Environment()  # 새 SimPy 환경으로 교체
        self.t = 0.0; self.completed = 0; self.W[:] = 0.0  # 현재 시간 t를 0초로 초기화, 완료된 작업 수 카운터 completed 초기화, 각 서버의 총 잔여 작업량 벡터 w를 모두 0으로 초기화
        self.arr_idx = 0; self.pending_job = False  # 다음에 라우팅할 도착 job의 인덱서 arr_idx를 0으로 리셋, 지금 라우팅 대기 중인 job이 있는가?에 대한 플래그를 False로 초기화
        inter = self.cfg.INTERARRIVAL_SEC  # 도착 간격(초)를 로컬 변수로 꺼내둠
        self.arrival_times = np.array([ inter*i for i in range(self.cfg.NUM_JOBS) ], dtype=np.float64)  # 총 NUM_JOBS개의 등간격 도착 시간 배열 생성([0, 60, 120, ...]초)
        self._advance_until(self.arrival_times[0])  # 첫 도착 시각까지 시간을 전진. 현재 스케줄은 첫 도착이 0초 이므로 delta=0 -> 첫 도착을 60초로 두고 싶으면 self._advance_until(self.arrival_times[1]); self.arr_idx = 1
        self.pending_job = True  # 첫 job이 도착해 있고 아직 라우팅되지 않았다로 표기 -> 이후 step(action)이 한 번 호출되어야 다음 도착으로 넘어갈 수 있음
        return self._observe()  # 현재 상태 관측값을 만들어 반환, 초기에는 거의 제로 벡터

    def step(self, action: int):  # 한 도착(epoch)에서 서버 선택(action)을 받아 시뮬레이션을 다음 결정 시점까지 진행하는 함수
        if not self.pending_job:
            raise RuntimeError("No pending job to route.")  # 현재 라우팅할 job이 없는데 step()이 호출되면 예외.
        obs = self.last_obs if self.last_obs is not None else self._observe()  # 보상 계산용으로 현재 관측값을 확보
        reward = self._reward(obs, int(action))  # 보상 계산
        self.W[int(action)] += self.cfg.PROC_TIME_SEC  # 선택한 서버의 총 남은 작업량 wk에 job 1개의 서비스 시간만큼 추가(해당 서버 큐에 job 한 개 더 쌓였다고 해석)
        self.pending_job = False  # 방금 도착한 job의 라우팅이 끝났으니 대기 job 없음으로 상태 전환
        self.arr_idx += 1  # 몇 번째 도착까지 처리했는지 인덱스 증가
        info = {"keys": self.KEYS, "action_routed_to": int(action)+1, "completed": self.completed}  # 진단/로그용 부가정보 -> action_rounted_to는 1~3, sompleted는 지금까지 완료된 job 수
        if self.arr_idx < self.cfg.NUM_JOBS:
            self._advance_until(float(self.arrival_times[self.arr_idx]))
            self.pending_job = True
            return self._observe(), reward, False, info  # 아직 다음 도착이 남아 있다면, 현재 시간에서 다음 도착 시각까지 전진하면서 서버들의 남은 작업량을 줄이고, 완료수 업데이트. 도착이 발생하였으므로 pending_job을 True로 바꾸고, 다음 의사결정 상태(done=False를 반환
        # 빈 상태까지
        if self.W.sum() <= 1e-12:
            return self._observe(), reward, True, info  # 더 이상 도착이 없을 때 분기: 시스템에 남은 일이 없다면 바로 종료(done=True)
        max_W = float(self.W.max()) 
        self._advance_until(self.t + max_W)
        return self._observe(), reward, True, info  # 아직 일이 남았으면, 가장 오래 걸리는 서버의 잔여 총 작업량만큼 시간을 전진하면 전체 시스템이 비게 됨. 그 시점의 관측을 반환하고 에피소드 종료(done=True)    

class JobRoutingGymEnv(gym.Env):  # Gym Environment 구현
    metadata = {"render.modes": ["human"]}  # Gymnasium 포맷의 환경 래퍼. render 모드 표기
    def __init__(self, config: Config|None=None):
        super().__init__()
        self.core = JobRoutingSimPyEnv(config)  # 내부에 위에 있는 SimPy 기반 코어 환경을 생성
        high = np.array([
            self.core.cfg.PROC_TIME_SEC,  # r1
            self.core.cfg.PROC_TIME_SEC,  # r2
            self.core.cfg.PROC_TIME_SEC,  # r3
            self.core.cfg.NUM_JOBS,       # q1
            self.core.cfg.NUM_JOBS,       # q2
            self.core.cfg.NUM_JOBS,       # q3
            self.core.cfg.NUM_JOBS,       # c
            self.core.cfg.NUM_JOBS * self.core.cfg.INTERARRIVAL_SEC 
            + self.core.cfg.NUM_JOBS * self.core.cfg.PROC_TIME_SEC  # t
        ], dtype=np.float32)

        low = np.zeros_like(high, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)  # 관측 공간 범위 정의

        self.action_space = spaces.Discrete(3)  # 행동 공간 정의
    def reset(self, *, SEED=None, options=None):  # Gymnasium 규격의 reset함수
        obs = self.core.reset(SEED=SEED if SEED is not None else None)
        return obs, {} 
    def step(self, action):  # 코어의 step 결과를 Gymnasium 포맷으로 변환하여 반환(여기서는 done을 곧바로 terminated로 씀)
        obs, reward, done, info = self.core.step(action)
        return obs, reward, done, False, info
    def render(self):  # 현재 상태를 간단히 출력
        print(f"t={self.core.t:.4f} s, obs={self.core.last_obs}, completed={self.core.completed}")
