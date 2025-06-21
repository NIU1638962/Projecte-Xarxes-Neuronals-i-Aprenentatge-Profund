# -*- coding: utf-8 -*- noqa
"""
Created on Thu May 29 04:42:59 2025

@author: JoelT
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# List all colormaps
print(plt.colormaps())

# OR visualize them
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack([gradient] * 10)

fig, axes = plt.subplots(nrows=20, figsize=(8, 20))
colormaps = plt.colormaps()[:20]  # Show first 20 cmaps

for ax, cmap_name in zip(axes, colormaps):
    ax.imshow(gradient, aspect='auto', cmap=cm.get_cmap(cmap_name))
    ax.set_title(cmap_name, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()
