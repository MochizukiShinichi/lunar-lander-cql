import gymnasium as gym
from stable_baselines3 import PPO
import d3rlpy
from d3rlpy.dataset import MDPDataset
import numpy as np
import os
import argparse

def collect_data(model, env_id, n_steps, is_random=False):
    env = gym.make(env_id)
    obs, info = env.reset()
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []

    for _ in range(n_steps):
        observations.append(obs)
        if is_random:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray) and action.size == 1:
                action = int(action.item())
        
        actions.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        terminals.append(terminated)
        timeouts.append(truncated)
        
        if terminated or truncated:
            obs, info = env.reset()
            
    # Force the last step to be a termination or timeout to satisfy d3rlpy if needed
    if not any(terminals) and not any(timeouts):
        timeouts[-1] = True
            
    dataset = MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=bool),
        timeouts=np.array(timeouts, dtype=bool)
    )
    env.close()
    return dataset

def main():
    env_id = "LunarLander-v3"
    
    # Check if we should retrain or load existing
    if not os.path.exists("medium_ppo.zip"):
        print("Training PPO from scratch...")
        env = gym.make(env_id)
        # Train a little bit for Medium Dataset (~50k steps)
        # 50k steps might result in around 0 to 50 average reward
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=50000)
        model.save("medium_ppo")
        
        # Train further for Expert Dataset (~450k more steps)
        # LunarLander usually solves at 200 reward around 300k-500k steps
        model.learn(total_timesteps=450000)
        model.save("expert_ppo")
        env.close()
    else:
        print("Loading existing PPO models...")

    print("Collecting Medium Dataset...")
    medium_model = PPO.load("medium_ppo")
    medium_dataset = collect_data(medium_model, env_id, n_steps=50000)
    medium_dataset.dump("medium_dataset.h5")

    print("Collecting Expert Dataset...")
    expert_model = PPO.load("expert_ppo")
    expert_dataset = collect_data(expert_model, env_id, n_steps=50000)
    expert_dataset.dump("expert_dataset.h5")

    print("Collecting Random Dataset...")
    random_dataset = collect_data(None, env_id, n_steps=50000, is_random=True)
    random_dataset.dump("random_dataset.h5")
        
    print("Data collection complete!")

if __name__ == "__main__":
    main()
