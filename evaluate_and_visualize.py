import gymnasium as gym
import numpy as np
import d3rlpy
import torch
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from d3rlpy.algos import DiscreteCQLConfig
import matplotlib.pyplot as plt
import pandas as pd
import os

# Monkeypatch torch.load for compatibility with PyTorch 2.6+ default weights_only=True
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

def evaluate_model(model_path, env_id, n_episodes=50):
    print(f"Evaluating {model_path}...")
    # Load model
    env = gym.make(env_id)
    cql = DiscreteCQLConfig().create(device="cpu")
    cql.build_with_env(env)
    cql.load_model(model_path)
    
    rewards = []
    successes = 0
    crashes = 0
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = cql.predict(np.expand_dims(obs, axis=0))[0]
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        rewards.append(ep_reward)
        if ep_reward >= 200:
            successes += 1
        if ep_reward <= -100: # Simple heuristic for crash, can be more precise if needed
            crashes += 1
            
    env.close()
    return np.mean(rewards), np.std(rewards), (successes / n_episodes) * 100, (crashes / n_episodes) * 100

def get_dataset_stats(dataset_path):
    print(f"Calculating stats for {dataset_path}...")
    from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
    with open(dataset_path, "rb") as f:
        dataset = ReplayBuffer.load(f, InfiniteBuffer())
    
    returns = []
    for episode in dataset.episodes:
        returns.append(np.sum(episode.rewards))
    
    return np.mean(returns)

def main():
    env_id = "LunarLander-v3"
    results = []
    
    comparisons = [
        ("expert_dataset.h5", "cql_expert.d3", "Expert"),
        ("medium_dataset.h5", "cql_medium.d3", "Medium"),
        ("random_dataset.h5", "cql_random.d3", "Random")
    ]
    
    for data_path, model_path, label in comparisons:
        if os.path.exists(model_path) and os.path.exists(data_path):
            ds_mean = get_dataset_stats(data_path)
            model_mean, model_std, success_rate, crash_rate = evaluate_model(model_path, env_id)
            
            results.append({
                "Label": label,
                "Dataset Mean": ds_mean,
                "Model Mean": model_mean,
                "Model Std": model_std,
                "Success Rate (%)": success_rate,
                "Crash Rate (%)": crash_rate
            })
        else:
            print(f"Skipping {label} - files not found.")
            
    if not results:
        print("No results to visualize!")
        return
        
    df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df.to_string(index=False))
    df.to_csv("performance_results.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df['Label']))
    width = 0.35
    
    plt.bar(x - width/2, df['Dataset Mean'], width, label='Dataset Mean (Demo)', alpha=0.7)
    plt.bar(x + width/2, df['Model Mean'], width, label='Model Mean (CQL)', alpha=0.7)
    
    plt.xlabel('Data Quality Level')
    plt.ylabel('Mean Cumulative Reward')
    plt.title('Performance Comparison: Dataset vs. Offline RL (CQL)')
    plt.xticks(x, df['Label'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("performance_comparison.png")
    print("\nPlot saved: performance_comparison.png")

    # Record simulation videos for the best models (Expert and Medium)
    from gymnasium.wrappers import RecordVideo
    for label in ["Expert", "Medium", "Random"]:
        model_file = f"cql_{label.lower()}.d3"
        if os.path.exists(model_file):
            print(f"Recording video for {label} agent...")
            env = gym.make(env_id, render_mode="rgb_array")
            cql = DiscreteCQLConfig().create(device="cpu")
            cql.build_with_env(env)
            cql.load_model(model_file)
            
            env = RecordVideo(env, video_folder="./videos", name_prefix=f"cql_{label.lower()}")
            
            obs, info = env.reset()
            done = False
            step_count = 0
            while not done:
                action = cql.predict(np.expand_dims(obs, axis=0))[0]
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Boundary check: Terminate if rocket goes too far off-screen
                # obs[0] is x-pos, obs[1] is y-pos
                off_screen = abs(obs[0]) > 1.1 or obs[1] > 1.2 or obs[1] < -0.1
                
                done = terminated or truncated or off_screen or step_count > 400
                step_count += 1
            env.close()

if __name__ == "__main__":
    main()
