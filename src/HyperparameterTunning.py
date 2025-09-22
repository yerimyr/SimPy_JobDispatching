import optuna
import torch
from environment import JobRoutingGymEnv, Config
from PPO import PPOAgent
from main_PPO_parallel import evaluate

def objective(trial):
    cfg = Config()
    cfg.LEARNING_RATE = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    cfg.GAMMA = trial.suggest_uniform("gamma", 0.90, 0.999)
    cfg.CLIP_EPSILON = trial.suggest_uniform("clip_epsilon", 0.01, 0.15)
    cfg.GAE_LAMBDA = trial.suggest_uniform("gae_lambda", 0.8, 0.99)
    cfg.ENT_COEF = trial.suggest_uniform("ent_coef", 0.0, 0.05)
    cfg.VF_COEF = trial.suggest_uniform("vf_coef", 0.1, 1.0)
    cfg.BATCH_SIZE = trial.suggest_categorical("batch_size", [16, 20, 32, 40])
    cfg.HIDDEN_SIZE = trial.suggest_categorical("hidden_size", [16, 32, 48, 64])

    env = JobRoutingGymEnv(cfg)
    eval_env = JobRoutingGymEnv(cfg)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dims=[n_actions],
        lr=cfg.LEARNING_RATE,
        gamma=cfg.GAMMA,
        clip_epsilon=cfg.CLIP_EPSILON,
        update_steps=cfg.UPDATE_STEPS,
        gae_lambda=cfg.GAE_LAMBDA,
        ent_coef=cfg.ENT_COEF,
        vf_coef=cfg.VF_COEF,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        hidden_size=cfg.HIDDEN_SIZE   
    )

    N_EPISODES = 2000   
    NUM_JOBS = cfg.NUM_JOBS
    total_timesteps = NUM_JOBS * N_EPISODES

    obs, _ = env.reset()
    for t in range(1, total_timesteps + 1):
        action_vec, log_prob = agent.select_action(obs)
        action = int(action_vec[0])
        obs2, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store_transition((obs, action_vec, reward, obs2, done, log_prob.item()))
        obs = obs2

        if done:
            agent.update()
            obs, _ = env.reset()

    mean_ret, std_ret = evaluate(agent, eval_env, n_episodes=5)
    return mean_ret  

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)  
    print("Best trial:", study.best_trial.params)
