import gymnasium as gym
import numpy as np
import d3rlpy
import torch
import os
import matplotlib.pyplot as plt
from d3rlpy.dataset import MDPDataset, ReplayBuffer, InfiniteBuffer
from d3rlpy.algos import DiscreteCQLConfig, DiscreteDecisionTransformerConfig, StatefulTransformerWrapper

# Patch torch for compatibility
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

def minimal_example():
    env_id = "LunarLander-v3"
    data_path = "minimal_test_data.h5"
    
    print("1. Collecting minimal data (2000 steps)...")
    env = gym.make(env_id)
    obs, _ = env.reset()
    observations, actions, rewards, terminals, timeouts = [], [], [], [], []
    for _ in range(2000):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        terminals.append(term)
        timeouts.append(trunc)
        obs = next_obs if not (term or trunc) else env.reset()[0]
    
    dataset = MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=bool),
        timeouts=np.array(timeouts, dtype=bool)
    )
    dataset.dump(data_path)
    env.close()

    print("2. Loading data...")
    with open(data_path, "rb") as f:
        buffer = ReplayBuffer.load(f, InfiniteBuffer())

    print("3. Training CQL model for 200 steps...")
    cql = DiscreteCQLConfig().create(device="cpu")
    cql.fit(buffer, n_steps=200, n_steps_per_epoch=100, show_progress=False)

    print("4. Training DT model for 200 steps...")
    dt = DiscreteDecisionTransformerConfig(context_size=10, max_timestep=1000).create(device="cpu")
    dt.fit(buffer, n_steps=200, n_steps_per_epoch=100, show_progress=False)

    print("5. Minimal Evaluation...")
    eval_env = gym.make(env_id)
    
    # Eval CQL
    cql.build_with_env(eval_env)
    obs, _ = eval_env.reset()
    action_cql = cql.predict(np.expand_dims(obs, axis=0))[0]
    print(f"CQL predicted action: {action_cql}")

    # Eval DT
    dt.build_with_env(eval_env)
    wrapper = StatefulTransformerWrapper(dt, target_return=200.0)
    obs, _ = eval_env.reset()
    action_dt = wrapper.predict(obs, 0.0)
    print(f"DT predicted action: {action_dt}")
    
    eval_env.close()
    print("Minimal pipeline for both CQL and DT successfully completed end-to-end!")

if __name__ == "__main__":
    minimal_example()
