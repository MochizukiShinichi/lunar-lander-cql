# Historical Comparison of Experiment Iterations

This document summarizes the evolution of experimental approaches and results in the Lunar Lander Offline RL project, as reflected in the commit history.

| Commit Hash | Date (Approx.) | Key Changes / Experiment Iteration                                     | Impact / Results Highlight                                          |
| :---------- | :------------- | :--------------------------------------------------------------------- | :------------------------------------------------------------------ |
| `5f0a70e`   | Initial        | **Initial Report Baseline:** Setup for "Lunar Lander Offline RL Data Comparison" | Established initial reporting structure and comparison goals.       |
| `9cd93f8`   | Subsequent     | **Reporting Refinement:** Added missing chart, cleaned video paths.    | Improved visualization and data presentation.                       |
| `16e4fb5`   | Subsequent     | **Detailed Setup:** Added Appendix with experiment setup details.      | Enhanced reproducibility and understanding of experimental conditions. |
| `9a0fca0`   | Subsequent     | **Decision Transformer Integration:** Implemented DT with GPU training and side-by-side rendering. | Introduction of a new, advanced RL approach; enabled visual comparison of agent performance. |
| `440368c`   | Subsequent     | **Medium Model Results Update:** Reflected true medium model results across datasets. | Provided updated metrics for medium-sized models, indicating performance on specific data scales. |
| `6ae9d97`   | Subsequent     | **Medium Dataset Metrics Refinement:** Rewrote report to frame true Medium dataset metrics appropriately. | Improved interpretation and presentation of medium dataset performance. |
| `e9c7c2c`   | Latest         | **Normalized DT Metrics Improvement:** Updated report to reflect improved normalized Decision Transformer metrics. | Showcased the latest performance enhancements for Decision Transformer models, particularly with normalization. |

**Summary of Evolution:**

The project began with an initial report setting the stage for offline RL data comparison. Early iterations focused on refining the reporting and visualization infrastructure. A significant pivot occurred with the integration of the **Decision Transformer** model, moving towards more advanced architectural approaches. Subsequent efforts concentrated on evaluating and refining the Decision Transformer's performance across different datasets, particularly the "medium" dataset, and improving the presentation and interpretation of its metrics, culminating in improved normalized metrics. The codebase reflects a progression from foundational reporting to the implementation and optimization of state-of-the-art offline reinforcement learning algorithms.