### COMPARISON PLOT ###
# Run this script after both experiment.py files have finished running.

import os
# Work from the script's own directory so all relative paths match the notebook
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

# Load pre-computed padded histories
padded_n = np.load("Results/histories_own_rul.npy")
padded_o = np.load("Results/histories_consultancy_rul.npy")

# Average best fitness per generation across all runs
avg_n = np.mean(padded_n, axis=0)
avg_o = np.mean(padded_o, axis=0)

# Comparison plot: average convergence
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(avg_n) + 1), avg_n, label="Own Predictions")
plt.plot(range(1, len(avg_o) + 1), avg_o, label="Consultancy Predictions")
plt.xlabel("Generation")
plt.ylabel("Average Best Penalty Cost")
plt.title("Task 2.3: Own vs Consultancy RUL Predictions")
plt.legend()
plt.tight_layout()
plt.savefig("Results/convergence_comparison.png")
plt.show()
print("Saved Results/convergence_comparison.png")
