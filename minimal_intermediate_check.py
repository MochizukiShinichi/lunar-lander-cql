import os
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteDecisionTransformerConfig, DiscreteCQLConfig
import d3rlpy

def collect_minimal_data(model, env_id, n_steps=5000):
    print(f"Collecting {n_steps} steps of data...")
    env = gym.make(env_id)
    observations, actions, rewards, terminals = [], [], [], []
    obs, info = env.reset()
    for _ in range(n_steps):
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
    
    if not os.path.exists("intermediate_ppo.zip"):
        print("Error: intermediate_ppo.zip not found. Please run the full script first to train the teacher.")
        return

    print("Loading PPO teacher...")
    model = PPO.load("intermediate_ppo", device="cpu") # Force CPU for faster collection
    
    dataset = collect_minimal_data(model, env_id)
    
    # Inspection for d3rlpy 2.x
    print(f"Dataset type: {type(dataset)}")
    # In d3rlpy 2.x, MDPDataset is a wrapper around a buffer
    print(f"Transitions: {dataset.transition_count}")
    # print(f"Episodes: {dataset.episode_count}") # Not available in compat v1
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Minimal CQL
    print("\n--- Testing CQL ---")
    cql = DiscreteCQLConfig().create(device=device)
    cql.fit(dataset, n_steps=500, n_steps_per_epoch=500, experiment_name="test_cql")
    cql.save_model("test_cql.d3")
    print("CQL test successful.")

    # Minimal DT
    print("\n--- Testing DT without Scaler ---")
    dt = DiscreteDecisionTransformerConfig(
        context_size=50, 
        max_timestep=1000,
        batch_size=64,
        learning_rate=1e-4
    ).create(device=device)
    
    # This is where the previous ValueError occurred
    dt.fit(dataset, n_steps=500, n_steps_per_epoch=500, experiment_name="test_dt")
    dt.save_model("test_dt.d3")
    print("DT test successful.")
    
    print("\nMinimal example working perfectly!")

if __name__ == "__main__":
    main()
