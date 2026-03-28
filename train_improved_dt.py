import os
import torch
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteDecisionTransformerConfig

def train_dt_improved(dataset_path, model_name, n_steps=30000):
    print(f"Loading dataset: {dataset_path}...")
    with open(dataset_path, "rb") as f:
        dataset = MDPDataset.load(f)
        
    print(f"Training Improved Decision Transformer on {dataset_path}...")
    # Apply 1) Context Window Expansion (context_size=50 instead of 20)
    # Apply 2) Return Scaling via 'standard' mathematical scaler (Normalizes rewards to have mean=0, std=1)
    dt = DiscreteDecisionTransformerConfig(
        context_size=50,
        max_timestep=1000,
        batch_size=64,
        learning_rate=1e-4,
        reward_scaler="standard" 
    ).create(device="cuda:0" if torch.cuda.is_available() else "cpu")
    
    dt.fit(dataset, n_steps=n_steps, n_steps_per_epoch=1000, experiment_name=model_name, show_progress=True)
    dt.save_model(model_name + ".d3")
    print(f"Saved {model_name}.d3")

datasets = [
    ("expert_dataset.h5", "dt_expert"),
    ("medium_dataset.h5", "dt_medium"),
    ("random_dataset.h5", "dt_random")
]

for data_path, model_name in datasets:
    if os.path.exists(data_path):
        # We overwrite the original dt_expert/dt_medium/dt_random .d3 models
        train_dt_improved(data_path, model_name, n_steps=30000)
    else:
        print(f"Error: {data_path} not found!")
