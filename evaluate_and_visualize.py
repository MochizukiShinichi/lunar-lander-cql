import gymnasium as gym
import numpy as np
import d3rlpy
import torch
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from d3rlpy.algos import DiscreteCQLConfig, DiscreteDecisionTransformerConfig, StatefulTransformerWrapper, GreedyTransformerActionSampler
import matplotlib.pyplot as plt
import pandas as pd
import os
from gymnasium.wrappers import RecordVideo

# Monkeypatch torch.load for compatibility with PyTorch 2.6+ default weights_only=True
import torch
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

def evaluate_model(model_path, env_id, algo_type="cql", n_episodes=50, label="Expert"):
    print(f"Evaluating {model_path} ({algo_type}) for {label}...")
    env = gym.make(env_id)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        if algo_type == "dt":
            # Try loading with different context sizes (Phase 4: 50, Phase 2: 20)
            model = None
            for cs in [50, 20]:
                try:
                    config = DiscreteDecisionTransformerConfig(
                        context_size=cs, 
                        max_timestep=1000,
                        batch_size=64,
                        learning_rate=1e-4
                    )
                    temp_model = config.create(device=device)
                    temp_model.build_with_env(env)
                    temp_model.load_model(model_path)
                    model = temp_model
                    print(f"Successfully loaded DT with context_size={cs}")
                    break
                except Exception:
                    continue
            
            if model is None:
                raise ValueError("Could not load DT model with any supported context_size")
            
            # Dynamic target return
            if label == "Expert": tr = 200.0
            elif label == "Intermediate": tr = 150.0
            elif label == "Medium": tr = 40.0
            else: tr = -100.0
            
            wrapper = StatefulTransformerWrapper(model, target_return=tr, action_sampler=GreedyTransformerActionSampler())
        else:
            config = DiscreteCQLConfig()
            model = config.create(device=device)
            model.build_with_env(env)
            model.load_model(model_path)
    except Exception as e:
        print(f"CRITICAL ERROR loading {model_path}: {e}")
        env.close()
        return np.nan, np.nan, np.nan, np.nan
    
    rewards = []
    successes = 0
    crashes = 0
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        reward = 0.0
        
        if algo_type == "dt":
            wrapper.reset()

        while not done:
            if algo_type == "dt":
                action = wrapper.predict(obs, reward)
            else:
                action = model.predict(np.expand_dims(obs, axis=0))[0]
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        rewards.append(ep_reward)
        if ep_reward >= 200:
            successes += 1
        if ep_reward <= -100:
            crashes += 1
            
    env.close()
    return np.mean(rewards), np.std(rewards), (successes / n_episodes) * 100, (crashes / n_episodes) * 100

import h5py

def get_dataset_stats(dataset_path):
    if not os.path.exists(dataset_path):
        return np.nan, np.nan
    print(f"Calculating stats for {dataset_path}...")
    
    try:
        # Try d3rlpy 2.x load_v1 first
        dataset = d3rlpy.dataset.load_v1(dataset_path)
        episode_rewards = []
        current_reward = 0
        rewards = dataset.rewards
        terminals = dataset.terminals
        for i in range(len(rewards)):
            current_reward += rewards[i]
            if terminals[i]:
                episode_rewards.append(current_reward)
                current_reward = 0
        
        successes = sum(1 for r in episode_rewards if r >= 200)
        success_rate = (successes / len(episode_rewards)) * 100 if episode_rewards else 0
        return np.mean(episode_rewards), success_rate
    except Exception:
        # Fallback to custom HDF5 format (observations_0, rewards_0, etc)
        print("Using custom HDF5 loader fallback...")
        with h5py.File(dataset_path, 'r') as f:
            reward_keys = [k for f_key in f.keys() if (k := f_key) and k.startswith('rewards_')]
            if not reward_keys:
                return np.nan, np.nan
            
            episode_returns = []
            for k in reward_keys:
                episode_returns.append(np.sum(f[k][()]))
            
            successes = sum(1 for r in episode_returns if r >= 200)
            success_rate = (successes / len(episode_returns)) * 100 if episode_returns else 0
            return np.mean(episode_returns), success_rate

def main():
    env_id = "LunarLander-v3"
    results = []
    
    levels = [
        ("expert", "Expert"),
        ("intermediate", "Intermediate"),
        ("medium", "Medium"),
        ("random", "Random")
    ]
    
    for suffix, label in levels:
        ds_path = f"{suffix}_dataset.h5"
        ds_mean = get_dataset_stats(ds_path)
        
        # Evaluate CQL
        cql_path = f"cql_{suffix}.d3"
        if os.path.exists(cql_path):
            cql_mean, _, cql_succ, _ = evaluate_model(cql_path, env_id, "cql", label=label)
        else:
            cql_mean, cql_succ = np.nan, np.nan
            
        # Evaluate DT
        dt_path = f"dt_{suffix}.d3"
        if os.path.exists(dt_path):
            dt_mean, _, dt_succ, _ = evaluate_model(dt_path, env_id, "dt", label=label)
        else:
            dt_mean, dt_succ = np.nan, np.nan
            
        results.append({
            "Label": label,
            "Dataset Mean": ds_mean,
            "CQL Mean": cql_mean,
            "DT Mean": dt_mean,
            "CQL Success (%)": cql_succ,
            "DT Success (%)": dt_succ
        })
            
    df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df.to_string(index=False))
    df.to_csv("performance_results.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(12, 7))
    x = np.arange(len(df['Label']))
    width = 0.25
    
    plt.bar(x - width, df['Dataset Mean'], width, label='Dataset Mean (Demo)', alpha=0.7)
    plt.bar(x, df['CQL Mean'], width, label='CQL Mean', alpha=0.7)
    plt.bar(x + width, df['DT Mean'], width, label='DT Mean', alpha=0.7)
    
    plt.xlabel('Data Quality Level')
    plt.ylabel('Mean Cumulative Reward')
    plt.title('Performance Comparison (with Intermediate Level)')
    plt.xticks(x, df['Label'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("performance_comparison.png")
    print("\nPlot saved: performance_comparison.png")

    # Record videos
    for suffix, label in levels:
        for algo, prefix in [("cql", "cql_"), ("dt", "dt_")]:
            model_file = f"{prefix}{suffix}.d3"
            if os.path.exists(model_file):
                print(f"Recording video for {algo.upper()} {label} agent...")
                env = gym.make(env_id, render_mode="rgb_array")
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                
                try:
                    if algo == "dt":
                        model = None
                        for cs in [50, 20]:
                            try:
                                config = DiscreteDecisionTransformerConfig(
                                    context_size=cs, 
                                    max_timestep=1000,
                                    batch_size=64,
                                    learning_rate=1e-4
                                )
                                temp_model = config.create(device=device)
                                temp_model.build_with_env(env)
                                temp_model.load_model(model_file)
                                model = temp_model
                                print(f"Successfully loaded DT for video with context_size={cs}")
                                break
                            except Exception:
                                continue
                        
                        if model is None:
                            raise ValueError(f"Could not load DT model {model_file} for video")
                        
                        if label == "Expert": tr = 200.0
                        elif label == "Intermediate": tr = 150.0
                        elif label == "Medium": tr = 40.0
                        else: tr = -100.0
                        wrapper = StatefulTransformerWrapper(model, target_return=tr, action_sampler=GreedyTransformerActionSampler())
                    else:
                        config = DiscreteCQLConfig()
                        model = config.create(device=device)
                        model.build_with_env(env)
                        model.load_model(model_file)
                except Exception as e:
                    print(f"Error loading model for video {model_file}: {e}")
                    env.close()
                    continue
                
                env = RecordVideo(env, video_folder="./videos", name_prefix=f"{prefix}{suffix}")
                
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
                    done = terminated or truncated or step_count > 1000
                    step_count += 1
                env.close()

if __name__ == "__main__":
    main()
