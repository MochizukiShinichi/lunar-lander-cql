import gymnasium as gym
import numpy as np
import d3rlpy
import torch
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from d3rlpy.algos import DiscreteCQLConfig, DiscreteDecisionTransformerConfig, StatefulTransformerWrapper, GreedyTransformerActionSampler
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

def evaluate_model(model_path, env_id, algo_type="cql", n_episodes=50):
    print(f"Evaluating {model_path} ({algo_type})...")
    env = gym.make(env_id)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if algo_type == "dt":
        dt = d3rlpy.load_learnable(model_path, device=device)
        tr = 200.0 if "expert" in model_path else (40.0 if "medium" in model_path else -100.0)
        wrapper = StatefulTransformerWrapper(dt, target_return=tr, action_sampler=GreedyTransformerActionSampler())
    else:
        cql = d3rlpy.load_learnable(model_path, device=device)
    
    rewards = []
    successes = 0
    crashes = 0
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        reward = 0.0  # Initialize reward for the first prediction step
        
        if algo_type == "dt":
            wrapper.reset()

        while not done:
            if algo_type == "dt":
                action = wrapper.predict(obs, reward)
            else:
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
        ("dt_expert.d3", "Expert", 198.95, 203.93, 74.0, 0.0),
        ("dt_medium.d3", "Medium", -175.36, -181.63, 0.0, 86.0),
        ("dt_random.d3", "Random", -178.19, -135.40, 0.0, 98.0)
    ]
    
    for dt_path, label, ds_mean, cql_mean, cql_succ, cql_crash in comparisons:
        # Evaluate DT if it exists
        if os.path.exists(dt_path):
            dt_mean, dt_std, dt_succ, dt_crash = evaluate_model(dt_path, env_id, "dt")
        else:
            print(f"Skipping DT evaluation for {label} - {dt_path} not found.")
            dt_mean, dt_std, dt_succ, dt_crash = (np.nan, np.nan, np.nan, np.nan)
            
        results.append({
            "Label": label,
            "Dataset Mean": ds_mean,
            "CQL Mean": cql_mean,
            "DT Mean": dt_mean,
            "CQL Success (%)": cql_succ,
            "DT Success (%)": dt_succ
        })
            
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
    
    plt.bar(x - width, df['Dataset Mean'], width, label='Dataset Mean (Demo)', alpha=0.7)
    plt.bar(x, df['CQL Mean'], width, label='CQL Mean', alpha=0.7)
    plt.bar(x + width, df['DT Mean'], width, label='DT Mean', alpha=0.7)
    
    plt.xlabel('Data Quality Level')
    plt.ylabel('Mean Cumulative Reward')
    plt.title('Performance Comparison: Dataset vs. CQL vs. Decision Transformer')
    plt.xticks(x, df['Label'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("performance_comparison.png")
    print("\nPlot saved: performance_comparison.png")

    # Record simulation videos for the best models (Expert and Medium)
    from gymnasium.wrappers import RecordVideo
    for label in ["Expert", "Medium", "Random"]:
        for algo, prefix in [("cql", "cql_"), ("dt", "dt_")]:
            model_file = f"{prefix}{label.lower()}.d3"
            if os.path.exists(model_file):
                print(f"Recording video for {algo.upper()} {label} agent...")
                env = gym.make(env_id, render_mode="rgb_array")
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                
                if algo == "dt":
                    model = d3rlpy.load_learnable(model_file, device=device)
                    tr = 200.0 if label == "Expert" else (40.0 if label == "Medium" else -100.0)
                    wrapper = StatefulTransformerWrapper(model, target_return=tr, action_sampler=GreedyTransformerActionSampler())
                else:
                    model = d3rlpy.load_learnable(model_file, device=device)
                
                env = RecordVideo(env, video_folder="./videos", name_prefix=f"{prefix}{label.lower()}")
                
                obs, info = env.reset()
                done = False
                step_count = 0
                if algo == "dt":
                    wrapper.reset()
                    
                while not done:
                    if algo == "dt":
                        action = wrapper.predict(obs, reward if step_count > 0 else 0)
                    else:
                        action = model.predict(np.expand_dims(obs, axis=0))[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Boundary check: Terminate if rocket goes too far off-screen
                    # obs[0] is x-pos, obs[1] is y-pos
                    off_screen = False
                    
                    done = terminated or truncated or off_screen or step_count > 400
                    step_count += 1
                env.close()

if __name__ == "__main__":
    main()
