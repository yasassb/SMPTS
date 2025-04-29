# Multi-Objective Genetic Algorithm for Shift Minimization Personnel Task Scheduling (SMPTS)

This project applies a **Multi-Objective Genetic Algorithm (NSGA-II)** to solve the **multi-skilled shift scheduling problem**, focusing on optimizing personnel task assignments under multiple conflicting objectives.

---

## 📌 Objectives

The algorithm optimizes shift scheduling by minimizing the following:

1. **Total number of shifts assigned**
2. **Shift overlapping for individual personnel**
3. **Violations of the maximum allowable shift overlap limit**

---

## 📁 Project Structure

```bash
.
├── data/                   # Input data files
│   └── ...                 
├── results/                # Output visualizations and results
│   ├── pareto_front_3d.png
│   ├── convergence_plot.png
│   ├── best_valid_gantt.png
│   └── best_valid_utilization.png
├── SMPTS.py                # Main code implementing NSGA-II for scheduling
├── data_extract.ipynb      # Notebook for preparing and extracting input data
└── README.md               # This documentation file