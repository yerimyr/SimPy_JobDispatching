from environment import JobRoutingGymEnv, Config

NUM_JOBS = 20
N_EPISODES = 1

def run_episode(env: JobRoutingGymEnv, episode_idx: int):
    print("=" * 60)
    print(f"[ Episode {episode_idx} start ]")
    print(f"- Total {env.core.cfg.NUM_JOBS} jobs arrived")
    print(f"- Number of servers: {env.core.cfg.NUM_SERVERS}, server processing time: {env.core.cfg.PROC_TIME_SEC} sec/job")
    print("=" * 60)

    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        step_count += 1

        # server1 -> server2 -> server3 -> ...
        action = (step_count - 1) % env.core.cfg.NUM_SERVERS  

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        r = obs[:3]   # remain time
        q = obs[3:6]  # length of queue
        completed = obs[6]
        t_now = obs[7]

        print(f"[Step {step_count}] Job routed → Server {action+1}")
        print(f" - Remaining times (rk): {r}")
        print(f" - Queue lengths (qk): {q}")
        print(f" - Completed jobs: {int(completed)}")
        print(f" - Current time: {t_now:.2f} sec/job")
        print("-" * 60)

    print(f"[ Episode {episode_idx} end ]")
    print(f"- Total Step: {step_count}")
    print(f"- Total Completed Jobs: {env.core.jobs_completed}")
    print(f"- Simulation Time: {env.core.env.now:.2f} sec/job")
    print("=" * 60 + "\n")


def main():
    cfg = Config()
    cfg.SEED = 42
    cfg.NUM_JOBS = 20

    env = JobRoutingGymEnv(cfg)

    for ep in range(1, N_EPISODES + 1):
        run_episode(env, ep)


if __name__ == "__main__":
    main()
