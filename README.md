# Lunar Lander Offline RL: CQL vs. Decision Transformer

This repository explores the performance and architectural trade-offs of two prominent Offline Reinforcement Learning (RL) algorithms—**Conservative Q-Learning (CQL)** and **Decision Transformer (DT)**—on the classic [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment.

We evaluate these models across four datasets of varying quality: **Expert**, **Intermediate**, **Medium**, and **Random**.

## 🚀 Overview

The primary goal of this study is to understand how different offline RL architectures handle varying levels of sub-optimal demonstration data. Our findings highlight a clear divide:
- **Decision Transformers** are powerful sequence models that excel at imitating high-quality trajectories but struggle with noise.
- **Conservative Q-Learning** is a value-based method that is significantly more robust at extracting signal from noisy, low-quality datasets.

## 📊 Key Results

| Dataset | Dataset Mean | CQL Success (%) | DT Success (%) |
| :--- | :--- | :--- | :--- |
| **Expert** | 275.5 | 98% | **100%** |
| **Intermediate** | 105.5 | 2% | **4%** |
| **Medium** | -123.6 | **1%** | 0% |
| **Random** | -205.8 | **1%** | 0% |

### Key Takeaways
1. **Sequence Modeling Advantage:** On high-quality data (Expert/Intermediate), DT consistently matches or exceeds teacher performance.
2. **Robustness of Q-Learning:** CQL remains more robust on low-quality data (Medium/Random), successfully "stitching" together productive actions where DT's trajectory-based imitation fails.
3. **DT Optimization:** Decision Transformer performance was significantly boosted by expanding the context window (size 50) and implementing reward normalization ("standard" scaling).

## 🛠️ Project Structure

- `train_cql.py`: Training script for the CQL baseline.
- `train_improved_dt.py`: Training script for the optimized Decision Transformer.
- `generate_data.py` / `generate_medium.py`: Utilities to generate the various offline datasets using PPO teachers.
- `evaluate_and_visualize.py`: Comprehensive evaluation pipeline that generates performance metrics and side-by-side comparison videos.
- `index.html`: A rich, interactive dashboard presenting the experimental findings, performance charts, and agent behavior videos.
- `experiment_comparison.md`: A historical log of the different experiment iterations and improvements.

## ⚙️ Installation

```bash
pip install torch d3rlpy gymnasium[box2d] stable-baselines3 pandas matplotlib
```

## 🏃 Quick Start

### 1. Generate Datasets
First, generate the offline datasets if they are not already present:
```bash
python generate_data.py
python generate_medium.py
```

### 2. Train Models
Train the CQL and DT models:
```bash
python train_true_medium.py
python train_improved_dt.py
```

### 3. Evaluate and Visualize
Generate the results dashboard and comparison videos:
```bash
python evaluate_and_visualize.py
```
Open `index.html` in your browser to view the full report.

## 📚 Acknowledgments
This project utilizes the excellent [d3rlpy](https://github.com/takuseno/d3rlpy) library for offline RL implementations and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for the LunarLander environment.
