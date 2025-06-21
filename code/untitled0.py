# -*- coding: utf-8 -*- noqa
"""
Created on Thu May 29 04:33:35 2025

@author: JoelT
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate a confusion matrix with most values near 0 and 2000
matrix_size = 10  # 10x10 matrix
conf_matrix = np.random.choice([0, 2000], size=(
    matrix_size, matrix_size), p=[0.55, 0.45])

# Add some random high values up to 35000
num_high_values = 5
high_indices = np.random.choice(
    matrix_size**2, size=num_high_values, replace=False)

for idx in high_indices:
    i, j = divmod(idx, matrix_size)
    conf_matrix[i, j] = np.random.randint(5000, 16000)

# Plot the matrix
plt.figure(figsize=(10, 8))

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="mako",
            xticklabels=["Pos", "Neg"] * (matrix_size // 2),
            yticklabels=["Pos", "Neg"] * (matrix_size // 2), robust=False)
plt.title("Synthetic Confusion Matrix")
plt.show()
