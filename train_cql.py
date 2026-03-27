import torch
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteDecisionTransformerConfig
import os

def train_dt_on_dataset(dataset_path, model_name, n_steps=50000):
    print(f"Loading dataset: {dataset_path}...")
    from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
    # d3rlpy 2.x load() expects a file handle and a buffer object
    with open(dataset_path, "rb") as f:
        dataset = ReplayBuffer.load(f, InfiniteBuffer())
    
    print(f"Training {model_name}...")
    dt = DiscreteDecisionTransformerConfig(
        context_size=20,
        max_timestep=1000
    ).create(device="cuda:0" if torch.cuda.is_available() else "cpu")
    
    dt.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=1000,
        experiment_name=model_name,
        show_progress=True
    )
    
    dt.save_model(f"{model_name}.d3")
    print(f"Model saved to {model_name}.d3")

def main():
    datasets = [
        ("expert_dataset.h5", "dt_expert"),
        ("medium_dataset.h5", "dt_medium"),
        ("random_dataset.h5", "dt_random")
    ]
    
    for data_path, model_name in datasets:
        if os.path.exists(data_path):
            train_dt_on_dataset(data_path, model_name, n_steps=30000)
        else:
            print(f"Error: {data_path} not found!")

if __name__ == "__main__":
    main()
