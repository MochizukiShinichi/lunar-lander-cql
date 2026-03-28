import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from d3rlpy.dataset import MDPDataset

def collect_data(model, env_id, n_steps=500000):
    env = gym.make(env_id)
    observations = []
    actions = []
    rewards = []
    terminals = []
    
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
    
    dataset = MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
    )
    return dataset

def main():
    env_id = "LunarLander-v3"
    env = gym.make(env_id)
    
    print("Training True Medium PPO Agent...")
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=20.0, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000, verbose=1)
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500000, callback=eval_callback)
    model.save("medium_ppo")
    env.close()
    
    print("Collecting True Medium Dataset...")
    medium_model = PPO.load("medium_ppo")
    medium_dataset = collect_data(medium_model, env_id, n_steps=500000)
    medium_dataset.dump("medium_dataset.h5")
    print("Medium Data collection complete!")

if __name__ == "__main__":
    main()
