import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.network_v1 import build_connectivity

if __name__ == "__main__":
    grid_size = 10
    w_exc, _, w_pv = build_connectivity(grid_size)

    # Excitatory colors
    colors_exc = ["darkslategrey", "white", "orangered"]
    exc_cmap = mcolors.LinearSegmentedColormap.from_list("exc_cmap", colors_exc, N=256)

    # Inhibitory colors
    color_positions = [0.0, 0.3, 0.7, 1.0]
    color_values = ["#00141e", "#86959B", "#C5D6D0", "#E9F2ED"]
    inh_cmap = mcolors.LinearSegmentedColormap.from_list("inh_cmap", list(zip(color_positions, color_values)), N=256)

    # Common plot parameters
    tick_size = 21
    label_size = 24
    line_width = 3

    plot_dir = "data/v1/connectivity"
    os.makedirs(plot_dir, exist_ok=True)

    # EXCITATORY CONNECTIVITY

    # Heatmap
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].imshow(w_exc, cmap="pink")
    ax[1].imshow(w_exc[50:250, 50:250], cmap="pink")
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[0].axis("off")
    ax[1].axis("off")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{plot_dir}/exc_heatmap.{ext}", dpi=500)

    # Weights distribution
    hist_values, bin_edges = np.histogram(w_exc[w_exc > 0], bins=20, density=True)
    bin_widths = np.diff(bin_edges)
    hist_values_proportion = hist_values * bin_widths

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(bin_edges[:-1], hist_values_proportion * 100, width=bin_widths, alpha=0.9, color='#bf8375', align="edge")
    ax.set_xlabel("Weight value", fontsize=label_size)
    ax.set_ylabel("% of weights", fontsize=label_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 56)

    for ext in ("png", "svg"):
        fig.savefig(f"{plot_dir}/exc_w_distr.{ext}")

    # Connection probabilities
    orientation_differences_probs = {0: 0.5, 22.5: 0.2, 45: 0.09, 67.5: 0.06, 90: 0.15}
    distance_probs = {0: 0.24, 1: 0.34, 2: 0.21, 3: 0.1, 4: 0.06, 5: 0.006, 6: 0.006, 7: 0.006, 8: 0.006, 9: 0.006}
    distance_probs_sum = sum(distance_probs.values())
    distance_probs = {k: v / distance_probs_sum for k, v in distance_probs.items()}  # Normalize

    n_samples = 100000  # Adjust sample size for smoother histograms
    orientation_samples = np.random.choice(list(orientation_differences_probs.keys()), size=n_samples,
                                           p=list(orientation_differences_probs.values()))
    distance_samples = np.random.choice(list(distance_probs.keys()), size=n_samples, p=list(distance_probs.values()))
    border_color = '#d27a59'
    fill_color = '#F7E5DD'

    fig = plt.figure(figsize=(10, 5), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.7])  # Left subplot narrower than right
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax2.hist(orientation_samples, bins=len(orientation_differences_probs),
             weights=np.ones_like(orientation_samples) / n_samples,  # Normalize to probability
             edgecolor=border_color, linewidth=line_width, facecolor=fill_color, histtype='stepfilled')
    ax2.set_xlabel("Orientation difference °", fontsize=label_size)
    ax2.set_ylabel("Connection probability", fontsize=label_size)
    ax2.set_xticks([0, 23, 45, 68, 90])
    ax2.set_ylim(0, 0.52)

    ax1.hist(distance_samples, bins=len(distance_probs),
             weights=np.ones_like(distance_samples) / n_samples,  # Normalize to probability
             edgecolor=border_color, linewidth=line_width, facecolor=fill_color, histtype='stepfilled')
    ax1.set_xlabel("Grid distance", fontsize=label_size)
    ax1.set_ylabel("Connection probability", fontsize=label_size)
    ax1.set_xticks(list(distance_probs.keys()))
    ax1.set_ylim(0, 0.52)

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{plot_dir}/exc_prob.{ext}")

    # INHIBITORY CONNECTIVITY

    # Heatmap
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    w = w_pv * (-1)
    ax[0].imshow(w, cmap=inh_cmap, vmin=0, vmax=5)
    ax[1].imshow(w[50:250, 50:250], cmap=inh_cmap, vmin=0, vmax=5)
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[0].axis("off")
    ax[1].axis("off")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{plot_dir}/inh_heatmap.{ext}", dpi=500)

    # Weights distribution
    hist_values, bin_edges = np.histogram(w_pv[w_pv < 0], bins=20, density=True)
    bin_widths = np.diff(bin_edges)
    hist_values_proportion = hist_values * bin_widths

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(bin_edges[:-1], hist_values_proportion * 100, width=bin_widths, alpha=0.9, color='darkslategrey',
           align="edge")
    ax.set_xlabel("Weight value", fontsize=24)
    ax.set_ylabel("% of weights", fontsize=24)
    ax.tick_params(axis='x', labelsize=21)
    ax.tick_params(axis='y', labelsize=21)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 56)
    # plt.title("Distribution of Connection Weights")
    plt.show()

    for ext in ("png", "svg"):
        fig.savefig(f"{plot_dir}/inh_w_distr.{ext}")

    # Connection probabilities
    # Define probabilities
    orientation_differences_probs = {0: 0.2, 22.5: 0.2, 45: 0.2, 67.5: 0.2, 90: 0.2}
    distance_probs = {0: 0.3, 1: 0.2, 2: 0.12, 3: 0.07, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    # Normalize distance probabilities
    distance_probs_sum = sum(distance_probs.values())
    distance_probs = {k: v / distance_probs_sum for k, v in distance_probs.items()}  # Normalize

    # Generate samples based on probabilities
    n_samples = 10_000  # Large sample size for smooth histograms
    orientation_samples = np.random.choice(list(orientation_differences_probs.keys()), size=n_samples,
                                           p=list(orientation_differences_probs.values()))
    distance_samples = np.random.choice(list(distance_probs.keys()), size=n_samples, p=list(distance_probs.values()))

    # Define aesthetics
    border_color = '#2F4F4F'  # Dark greenish-gray
    fill_color = '#D5DBDB'  # Soft gray

    fig = plt.figure(figsize=(10, 5), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.7])  # Adjust subplot proportions
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    orientation_bins = np.array([0, 22.5, 45, 67.5, 90])  # Keep exact orientation values
    distance_bins = np.arange(-0.5, 10.5, 1)  # Ensure bins are properly spaced

    ax1.hist(distance_samples, bins=distance_bins,
             weights=np.ones_like(distance_samples) / n_samples,  # Normalize to probability
             edgecolor=border_color, linewidth=line_width, facecolor=fill_color, histtype='stepfilled')
    ax1.set_xlabel("Grid distance", fontsize=label_size)
    ax1.set_ylabel("Connection probability", fontsize=label_size)
    ax1.set_xticks(np.arange(10))  # Ensure correct integer tick placement
    ax1.set_ylim(0, 0.52)

    ax2.hist(orientation_samples, bins=np.append(orientation_bins - 11.25, 101.25),  # Adjust bin edges to center bars
             weights=np.ones_like(orientation_samples) / n_samples,  # Normalize to probability
             edgecolor=border_color, linewidth=line_width, facecolor=fill_color, histtype='stepfilled')
    ax2.set_xlabel("Orientation difference °", fontsize=label_size)
    ax2.set_ylabel("Connection probability", fontsize=label_size)
    ax2.set_xticks([0, 23, 45, 68, 90])
    ax2.set_ylim(0, 0.52)

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{plot_dir}/inh_prob.{ext}", bbox_inches='tight')