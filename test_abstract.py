import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import multiprocessing as mp
import seaborn as sns

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.config import params_fixed
from src.utils import build_input_one_to_one, generate_lgn_inputs
from src.network_abstract import build_connectivity
from src.model import NetworkLIF
from src.measure import measure_rsync

np.set_printoptions(suppress=True)

from scipy.stats import ks_2samp, entropy
from scipy.stats import gaussian_kde

def bc_coefficient(sample1, sample2):
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)

    sample1 /= np.sum(sample1)
    sample2 /= np.sum(sample2)

    return np.sum(np.sqrt(sample1 * sample2))

def kolmogorov_smirnov_test(sample1, sample2):
    """
    Perform KS test to check if two samples come from the same distribution.
    Returns the KS statistic and p-value.
    """
    ks_stat, p_value = ks_2samp(sample1, sample2)
    return ks_stat, p_value

def symmetric_kl_divergence(sample1, sample2, bandwidth=0.1, epsilon=1e-10):
    """
    Compute symmetric KL divergence using KDE for density estimation, avoiding infinities.
    """
    kde1 = gaussian_kde(sample1, bw_method=bandwidth)
    kde2 = gaussian_kde(sample2, bw_method=bandwidth)

    x_vals = np.linspace(min(min(sample1), min(sample2)), max(max(sample1), max(sample2)), 1000)
    p = kde1(x_vals)
    q = kde2(x_vals)

    # Avoid zero probabilities by adding a small constant (Laplace smoothing)
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)

    # Normalize to ensure proper probability distributions
    p /= np.sum(p)
    q /= np.sum(q)

    kl_pq = entropy(p, q)
    kl_qp = entropy(q, p)

    return kl_pq + kl_qp


def run_simulation(rate_var, rate_mean, model_class, model_params, patterns, n_neurons, n_sources,
                   pattern_active, length, w_exc, w_inh, w_ff, queue):
    n_patterns = len(patterns)
    min_rate, max_rate = rate_mean - rate_var, rate_mean + rate_var
    input_rates = range(min_rate, max_rate + 1, 1)

    n_samples = 160
    iterations = max(1, n_samples // len(input_rates))

    x_sc = []
    x_rsync = []
    y = []

    for input_rate in input_rates:
        print(rate_var, input_rate, "/", max_rate)

        for _ in range(iterations):
            for fam in (True, False):
                model = model_class(n_neurons=n_neurons, n_inputs=n_neurons,
                                    w_exc=w_exc, w_inh=w_inh, w_ff=w_ff, **model_params)

                if fam:
                    pattern_orig = patterns[np.random.choice(n_patterns)]
                    pattern_cur = sorted(np.random.choice(pattern_orig, pattern_active, replace=False))
                else:
                    pattern_orig = sorted(np.random.choice(np.arange(n_neurons), pattern_active, replace=False))
                    pattern_cur = pattern_orig.copy()

                poisson_input = generate_lgn_inputs(n_neurons, n_sources, pattern_cur, input_rate, length + 1)
                _, spikes = model.simulate(length=length, external_input=poisson_input)

                x_sc.append(spikes[pattern_cur].sum(1))
                x_rsync.append(measure_rsync(spikes[pattern_orig]))
                y.append(bool(fam))

    x_rsync = np.array(x_rsync)
    x_sc = np.array([np.mean(el) for el in x_sc])
    y = np.array(y)
    print("DONE", rate_var, min_rate, max_rate, len(y))

    ids_fam = np.where(y)[0]
    ids_new = np.where(~y)[0]

    ks_stat_rsync, ks_p_rsync = kolmogorov_smirnov_test(x_rsync[ids_fam], x_rsync[ids_new])
    kl_rsync = symmetric_kl_divergence(x_rsync[ids_fam], x_rsync[ids_new])

    ks_stat_sc, ks_p_sc = kolmogorov_smirnov_test(x_sc[ids_fam], x_sc[ids_new])
    kl_sc = symmetric_kl_divergence(x_sc[ids_fam], x_sc[ids_new])

    # Send results back to main process
    results = {
        "rate_var": rate_var,  "x_rsync": x_rsync, "x_sc": x_sc, "y": y,
        "ks_stat_rsync": ks_stat_rsync, "ks_p_rsync": ks_p_rsync, "ks_stat_sc": ks_stat_sc, "ks_p_sc": ks_p_sc,
        "kl_rsync": kl_rsync, "kl_sc": kl_sc
    }
    queue.put(results)

if __name__ == "__main__":
    length = 500
    model_class = NetworkLIF
    rate_mean = 50
    n_sources = 1

    # parameters: CONNECTIVITY
    w_ff_val = 3
    pattern_size = 100
    n_patterns = 40
    n_neurons = 1000
    w_exc_intra_cluster_ratio = 0.5

    # parameters: EXPERIMENT
    pattern_active_mask = 0.0

    input_size = int(pattern_size * 1.0)
    pattern_active = int(input_size * (1 - pattern_active_mask))

    model_params = params_fixed.copy()

    # build connectivity
    w_ff = build_input_one_to_one(n_neurons=n_neurons, n_inputs=n_neurons) * w_ff_val
    patterns, w_exc, w_som, w_pv = build_connectivity(n_neurons=n_neurons,
                                                      n_patterns=n_patterns,
                                                      pattern_size=pattern_size,
                                                      w_exc_intra_cluster_ratio=w_exc_intra_cluster_ratio)
    w_inh = (w_som + w_pv)
    #w_exc *= 0.0

    print("w_exc", w_exc.mean(axis=1).mean(), w_exc.sum(axis=1).mean())
    print("w_inh", w_inh.mean(axis=1).mean(), w_inh.sum(axis=1).mean())

    #plt.imshow(w_exc)
    #plt.show()

    rate_list = [0, 5, 10, 15, 20, 25, 30, 35, 40]

    ks_stat_dict = {"sc": [], "rsync": []}
    ks_p_dict = {"sc": [], "rsync": []}
    kl_dict = {"sc": [], "rsync": []}

    fig, ax = plt.subplots(figsize=(20, 7), nrows=2, ncols=5)
    plot_dir = f"plots/abstract/{n_patterns}"
    os.makedirs(plot_dir, exist_ok=True)

    manager = mp.Manager()
    queue = manager.Queue()
    processes = []
    results = {"rate_var": [],
               "ks_stat_rsync": [], "ks_p_rsync": [], "ks_stat_sc": [], "ks_p_sc": [],
               "kl_rsync": [], "kl_sc": []}
    results_dist = {}

    for rate_var in rate_list:
        p = mp.Process(target=run_simulation, args=(rate_var, rate_mean, model_class, model_params,
                                                    patterns, n_neurons, n_sources, pattern_active,
                                                    length, w_exc, w_inh, w_ff, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    while not queue.empty():
        result_p = queue.get()
        for key in results:
            results[key].append(result_p[key])

        results_dist[result_p["rate_var"]] = (result_p["x_rsync"], result_p["x_sc"], result_p["y"])

    # Sort results
    df = pd.DataFrame(results)
    df = df.sort_values(by=["rate_var"], ascending=True)
    df.to_csv(f"{plot_dir}/dist_stat.csv", index=False)

    print("ALL DONE, START PLOTTING")
    # Create distributions plot
    fig, ax = plt.subplots(figsize=(20, 7), nrows=2, ncols=5)

    for rate_var in rate_list:
        print("plot", rate_var)
        x_rsync, x_sc, y = results_dist[rate_var]
        ids_fam = np.where(y)[0]
        ids_new = np.where(~y)[0]

        if rate_var % 2 == 0:
            sns.kdeplot(x_rsync[ids_fam], fill=True, color="orangered", alpha=0.25, linewidths=2,
                            label="Familiar (pattern)", ax=ax[0, rate_var // 10])
            sns.kdeplot(x_rsync[ids_new], fill=True, color="coral", alpha=0.15, linewidths=2, linestyles="dashed",
                            label="Distractor", ax=ax[0, rate_var // 10])
            ax[0, rate_var // 10].set_title(f"Variability {rate_var}", fontsize=22)

            sns.kdeplot(x_sc[ids_fam], fill=True, color="teal", alpha=0.25, linewidths=2,
                            label="Familiar (pattern)", ax=ax[1, rate_var // 10])
            sns.kdeplot(x_sc[ids_new], fill=True, color="cadetblue", alpha=0.15, linewidths=2, linestyles="dashed",
                            label="Distractor", ax=ax[1, rate_var // 10])

    for i_col in range(5):
        ax[0, i_col].spines['top'].set_visible(False)
        ax[0, i_col].spines['right'].set_visible(False)
        ax[0, i_col].set_ylim(0, 220)
        ax[0, i_col].set_xlim(0, 0.06)
        ax[0, i_col].tick_params(axis='x', labelsize=20)
        ax[0, i_col].tick_params(axis='y', labelsize=20)
        ax[0, i_col].set_ylabel(" ")

        ax[1, i_col].spines['top'].set_visible(False)
        ax[1, i_col].spines['right'].set_visible(False)
        ax[1, i_col].set_ylim(0, 1.9)
        ax[1, i_col].set_xlim(0, 13)
        ax[1, i_col].tick_params(axis='x', labelsize=20)
        ax[1, i_col].tick_params(axis='y', labelsize=20)
        ax[1, i_col].set_ylabel(" ")

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/distributions.png")
    fig.savefig(f"{plot_dir}/distributions.svg")

    # Create summary plot
    fig, ax = plt.subplots(figsize=(8, 4), nrows=1, ncols=2)

    ax[0].plot(rate_list, df["ks_stat_rsync"], color="orangered", label="Rsync")
    ax[0].plot(rate_list, df["ks_stat_sc"], color="teal", label="Spike count")
    ax[0].set_title("KS statistic", fontsize=20)
    ax[0].set_xlabel("Input rate variability", fontsize=18)
    ax[0].tick_params(axis='x', labelsize=16)
    ax[0].tick_params(axis='y', labelsize=16)
    ax[0].legend(fontsize=16)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ax[1].plot(rate_list, df["kl_rsync"], color="orangered", label="Rsync")
    ax[1].plot(rate_list, df["kl_sc"], color="teal", label="Spike count")
    ax[1].set_title("Symm. KL-divergence", fontsize=20)
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    ax[1].set_xlabel("Input rate variability", fontsize=18)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/summary_dist.png")
    fig.savefig(f"{plot_dir}/summary_dist.svg")

    print("RSYNC")
    print(df)