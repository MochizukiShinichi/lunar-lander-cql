import os
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteDecisionTransformerConfig, DiscreteCQLConfig
import d3rlpy

def collect_data(model, env_id, n_steps=500000):
    env = gym.make(env_id)
    observations, actions, rewards, terminals = [], [], [], []
    obs, info = env.reset()
    for step in range(n_steps):
        if step % 50000 == 0:
            print(f"Collecting step {step}/{n_steps}...")
        action, _states = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        terminals.append(terminated or truncated)
        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    return MDPDataset(observations=np.array(observations), actions=np.array(actions),
                      rewards=np.array(rewards), terminals=np.array(terminals))

def main():
    env_id = "LunarLander-v3"
    env = gym.make(env_id)
    
    if os.path.exists("intermediate_ppo.zip"):
        print("Loading existing Intermediate PPO Agent (intermediate_ppo.zip)...")
        model = PPO.load("intermediate_ppo")
    else:
        print("Training Intermediate PPO Agent (Stopping at ~100 average reward)...")
        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100.0, verbose=1)
        eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000, verbose=1)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=500000, callback=eval_callback)
        model.save("intermediate_ppo")
    env.close()
    
    if os.path.exists("intermediate_dataset.h5"):
        print("Loading existing Intermediate Dataset (intermediate_dataset.h5)...")
        try:
            dataset = d3rlpy.dataset.load_v1("intermediate_dataset.h5")
            print(f"Dataset loaded: {dataset.transition_count} transitions.")
        except Exception as e:
            print(f"Dataset corrupted or incompatible: {e}. Regenerating...")
            dataset = collect_data(model, env_id, n_steps=500000)
            dataset.dump("intermediate_dataset.h5")
    else:
        print("Collecting Intermediate Dataset...")
        dataset = collect_data(model, env_id, n_steps=500000)
        dataset.dump("intermediate_dataset.h5")
    
    print("Training models on Intermediate Dataset...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("Training CQL...")
    if not os.path.exists("cql_intermediate.d3"):
        cql = DiscreteCQLConfig().create(device=device)
        cql.fit(dataset, n_steps=30000, n_steps_per_epoch=1000, experiment_name="cql_intermediate", show_progress=True)
        cql.save_model("cql_intermediate.d3")
    
    print("Training DT (Optimized)...")
    if not os.path.exists("dt_intermediate.d3"):
        # Note: reward_scaler removed due to d3rlpy 2.x trajectory slicer bug
        dt = DiscreteDecisionTransformerConfig(
            context_size=50, 
            max_timestep=1000,
            batch_size=64,
            learning_rate=1e-4
        ).create(device=device)
        dt.fit(dataset, n_steps=30000, n_steps_per_epoch=1000, experiment_name="dt_intermediate", show_progress=True)
        dt.save_model("dt_intermediate.d3")
    print("Done!")

if __name__ == "__main__":
    main()
