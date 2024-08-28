import os
import numpy as np
import matplotlib.pyplot as plt
import util

def plot_history(num_steps, history):
    cols = min(10, num_steps)
    rows = (num_steps-1) // cols + 1
    fig, axs = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    for i in range(num_steps):
        if num_steps == 1:
            ax = axs
        elif rows == 1:
            ax = axs[i]
        else:
            ax = axs[i // cols, i % cols]
        im = ax.imshow(history[i], vmin=min([np.min(mat) for mat in history] + [-0.4]), vmax=max([np.max(mat) for mat in history] + [1]), cmap='hsv_r')
        ax.set_title(f"t={i+1}")
        fig.colorbar(im, ax=ax, orientation='vertical')
    plt.tight_layout()
    plt.savefig(os.path.join(util.root(), "plot.png"))


