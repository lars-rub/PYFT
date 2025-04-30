import os
import numpy as np
import matplotlib.pyplot as plt
from src import util
from matplotlib import colors
import time
from src.time_message import tprint

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),   
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_history(num_ticks, history, save_plot, step_names):
    if len(history) == 0:
        print("Nothing to plot")
        return
    cmap = truncate_colormap(plt.get_cmap('hsv_r'), 0.29, 1)
    
    num_steps = len(history[0])
    cols = min(10, num_ticks) + 1
    rows_per_step = ((num_ticks-1) // (cols-1) + 1)
    rows = rows_per_step * num_steps

    fig = plt.figure(figsize=(2*cols, 2*rows))
    for step_idx in range(num_steps):
        ax_label = fig.add_subplot(rows, cols, (step_idx * rows_per_step) * cols + 1)
        label = step_names[step_idx]
        if len(label) > 8:
            label = label.replace(".", ".\n")
        ax_label.text(0.5, 0.5, label, fontsize=12, ha='center')
        ax_label.set_axis_off()
        colorbar_displayed = False

        if not np.any([step_mats[step_idx] is not None for step_mats in history]):
            continue
        step_history_notna = [step_mats[step_idx] for step_mats in history if not step_mats[step_idx] is None]
        vmin = min([np.min(mat) for mat in step_history_notna])
        vmax = max([np.max(mat) for mat in step_history_notna])

        for i in range((cols - 1) * rows_per_step):
            if i >= num_ticks:
                ax = fig.add_subplot(rows, cols, step_idx * cols + i + 2)
                ax.set_axis_off()
                continue
            row = (i // (cols-1) + step_idx * rows_per_step)
            col = i % (cols-1)
            dimensionality = len(history[i][step_idx].shape)
            if dimensionality >= 3:
                ax = fig.add_subplot(rows, cols, row * cols + col + 2, projection='3d')
            else:
                ax = fig.add_subplot(rows, cols, row * cols + col + 2)
                
            # Plot data
            data = history[i][step_idx]
            im = None
            if dimensionality > 3:
                # reduce dimensionality to 3
                # sum last axis until its 3d
                while len(data.shape) > 3:
                    data = np.sum(data, axis=-1)
            if dimensionality == 1:
                ax.plot(data)
            elif dimensionality == 2:
                im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
            elif dimensionality >= 3:
                # Let markersize depend on the value of the matrix
                markersize_min = 2
                markersizes = markersize_min + 20 * data
                markersizes = markersizes.at[markersizes < 0].set(markersize_min)
                im = ax.scatter(*np.where(data > np.min(data) - 1), c=data, s = markersizes, vmin=vmin, vmax=vmax, cmap=cmap)
            if step_idx == 0:
                ax.set_title(f"t={i+1}")
            if im is not None and not colorbar_displayed:
                colorbar_displayed = True
                fig.colorbar(im, ax=ax_label, orientation='vertical')
    plt.tight_layout()
    if save_plot:
        folder = os.path.join(util.root(), "output")
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, f"plot_{int(time.time())}.png"))
        tprint("Plot done")
    else:
        plt.show()


