import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from matplotlib.colors import to_rgba

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.config import params_fixed
from src.utils import build_input_one_to_one, generate_lgn_inputs
from src.network_abstract import build_connectivity
from src.model import NetworkLIF
from src.measure import measure_rsync, measure_mean_sc

np.set_printoptions(suppress=True)
rs = 42

def fam_barplot(performances, plot_name):
    # Example performance scores for 7 models (e.g., accuracy, F1-score, etc.)
    models = ["I", "M", "A",
              "I", "M", "A",
              ""]
    performance = list(performances) + [0.33]

    # Define colors and transparency levels
    base_colors = ["teal", "teal", "teal", "orangered", "orangered", "orangered", "black"]
    alphas = [0.65, 0.35, 0.15, 0.65, 0.35, 0.15, 0.2]  # Adjust transparency

    # Convert base colors into RGBA format with different transparencies
    colors = [to_rgba(color, alpha) for color, alpha in zip(base_colors, alphas)]

    # Define group positions (grouping green, orange, and grey separately)
    group_positions = [0, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0]  # Keeping groups together with spacing
    group_names = ["Spike count", "Rsync", "Baseline"]
    group_x_positions = [0.25, 1.25, 2.1]

    # Create bar plot with custom width and grouped bars
    fig, ax = plt.subplots(figsize=(6, 5))
    bar_width = 0.2  # Adjust bar width

    bars = ax.bar(group_positions, performance, color=colors, edgecolor=base_colors,
                  linewidth=1.25, width=bar_width)

    ax.axhline(y=performance[-1], color="black", linestyle="dashed", linewidth=1.5, label="Model 7 Threshold")

    # Add group names below the x-ticks, centered under each group
    for pos, group_name in zip(group_x_positions, group_names):
        ax.text(pos, -0.175, group_name, ha="center", fontsize=20, fontweight="bold")

    # Set x-axis labels correctly under grouped bars
    ax.set_xticks(group_positions, models)  # rotation=40, ha="right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # Labels and title
    ax.set_ylabel("F1 Score", fontsize=24)
    ax.set_ylim(0, 1.1)

    fig.savefig(plot_name)

def process_ratio(w_exc_intra_cluster_ratio, queue, model_class, model_params, n_neurons, w_ff,
                  length, input_rates, iterations, n_patterns, pattern_active, pattern_size, n_sources, plot_dir):
    """Worker function to process a single w_exc_intra_cluster_ratio."""
    results = {"f1": {
                    "Ratio": w_exc_intra_cluster_ratio,
                    "Rsync Input": 0, "Rsync Max Rsync": 0, "Rsync Max SC": 0, "Rsync Total": 0,
                    "SC Input": 0, "SC Max Rsync": 0, "SC Max SC": 0, "SC Total": 0,
                    },
              "fp_fn": {
                    "Ratio": w_exc_intra_cluster_ratio,
                    "Rsync Input": 0, "Rsync Max Rsync": 0, "Rsync Max SC": 0, "Rsync Total": 0,
                    "SC Input": 0, "SC Max Rsync": 0, "SC Max SC": 0, "SC Total": 0,
                    },
              "class_f1": {
                    "Ratio": w_exc_intra_cluster_ratio,
                    "Max pattern Rsync": 0, "Max pattern SC": 0,
                    "Overlap Familiar": 0,
                    "Overlap Novel": 0
                    }
                }

    # build connectivity
    patterns, w_exc, w_som, w_pv = build_connectivity(n_neurons=n_neurons,
                                                      n_patterns=n_patterns,
                                                      pattern_size=pattern_size,
                                                      w_exc_intra_cluster_ratio=w_exc_intra_cluster_ratio)
    w_inh = (w_som + w_pv)
    #w_exc *= 0.0

    print("w_exc", w_exc.mean(axis=1).mean(), w_exc.max(axis=1).mean())
    print("w_inh", w_inh.mean(axis=1).mean(), w_inh.min(axis=1).mean())

    y_class_sc, y_class_rsync, y_class, y_fam = [], [], [], []
    x_rsync_input, x_rsync_max_sc, x_rsync_max_rsync, x_rsync_all = [], [], [], []
    x_sc_input, x_sc_max_sc, x_sc_max_rsync, x_sc_all = [], [], [], []
    sc_overlap_new, sc_overlap_fam = [], []

    model = model_class(n_neurons=n_neurons, n_inputs=n_neurons,
                        w_exc=w_exc, w_inh=w_inh, w_ff=w_ff,
                        **model_params)

    for input_rate in input_rates:
        print(input_rate)
        for iteration in range(iterations):
            pattern_new = sorted(np.random.choice(np.arange(n_neurons), pattern_active, replace=False))
            pattern_idx = np.random.randint(0, n_patterns)
            pattern_fam = patterns[pattern_idx]
            pattern_fam_cur = sorted(np.random.choice(pattern_fam, pattern_active, replace=False))

            patterns_all = patterns + [pattern_new]

            for sample_fam, sample_input in {0: pattern_new, 1: pattern_fam_cur}.items():
                poisson_input = generate_lgn_inputs(n_neurons, n_sources, sample_input, input_rate, length + 1)
                voltage, spikes = model.simulate(length=length, external_input=poisson_input)

                top_100_ids = np.argsort(spikes.sum(1))[-100:][::-1]
                overlap = len(np.intersect1d(top_100_ids, sample_input)) / 100
                if sample_fam == 0:
                    sc_overlap_new.append(overlap)
                else:
                    sc_overlap_fam.append(overlap)

                sample_sc = [measure_mean_sc(spikes[p]) for p in patterns] + [measure_mean_sc(spikes[pattern_new])]
                sample_rsync = [measure_rsync(spikes[p]) for p in patterns] + [measure_rsync(spikes[pattern_new])]

                max_sc_p = patterns_all[np.argmax(sample_sc)]
                max_rsync_p = patterns_all[np.argmax(sample_rsync)]

                sample_class = int(pattern_idx) if sample_fam == 1 else int(n_patterns)

                x_rsync_input.append(measure_rsync(spikes[sample_input]))
                x_rsync_max_rsync.append(measure_rsync(spikes[max_rsync_p]))
                x_rsync_max_sc.append(measure_rsync(spikes[max_sc_p]))
                x_rsync_all.append(measure_rsync(spikes))

                x_sc_input.append(measure_mean_sc(spikes[sample_input]))
                x_sc_max_rsync.append(measure_mean_sc(spikes[max_rsync_p]))
                x_sc_max_sc.append(measure_mean_sc(spikes[max_sc_p]))
                x_sc_all.append(measure_mean_sc(spikes))

                y_class_sc.append(np.argmax(sample_sc))
                y_class_rsync.append(np.argmax(sample_rsync))
                y_class.append(sample_class)
                y_fam.append(sample_fam)

    baseline_class = np.random.choice(np.arange(1, n_patterns + 2), size=len(y_class))
    baseline_fam = np.random.choice([0, 1], size=len(y_fam))

    print("PATTERN CLASSIFICATION")
    print("Rsync", accuracy_score(y_class, y_class_rsync), f1_score(y_class, y_class_rsync, average="weighted"))
    print("Spike count", accuracy_score(y_class, y_class_sc), f1_score(y_class, y_class_sc, average="weighted"))
    print("Baseline", accuracy_score(y_class, baseline_class), f1_score(y_class, baseline_class, average="weighted"))

    results["class_f1"]["Max pattern Rsync"] = f1_score(y_class, y_class_rsync, average="weighted")
    results["class_f1"]["Max pattern SC"] = f1_score(y_class, y_class_sc, average="weighted")

    results["class_f1"]["Overlap Familiar"] = np.mean(sc_overlap_fam)
    results["class_f1"]["Overlap Novel"] = np.mean(sc_overlap_new)

    print("FAMILIARITY")
    performances = []
    for x_key, x_val in {"SC Input": x_sc_input, "SC Max SC": x_sc_max_sc, "SC Max Rsync": x_sc_max_rsync, "Total SC": x_sc_all,
                         "Rsync Input": x_rsync_input, "Rsync Max SC": x_rsync_max_sc, "Rsync Max Rsync": x_rsync_max_rsync, "Total Rsync": x_rsync_all,
                         }.items():

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        f1_scores = []
        fp_fn_ratios = []

        X = np.array(x_val).reshape(-1, 1)
        y = np.array(y_fam)

        for train_index, val_index in skf.split(X, y):  # Stratified splits
            # Split into training and validation sets
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Train model using F1-score
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            y_val_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_val_pred)
            f1_scores.append(f1)

            # Evaluate on validation set using FPR - FNR
            fp = np.sum((y_val_pred == 1) & (y_val == 0))
            fn = np.sum((y_val_pred == 0) & (y_val == 1))
            tp = np.sum((y_val_pred == 1) & (y_val == 1))
            tn = np.sum((y_val_pred == 0) & (y_val == 0))
            fp_fn_ratio = fp / (fp + tn + 0.0001) - fn / (fn + tp + 0.0001)
            fp_fn_ratios.append(fp_fn_ratio)

            print(x_key, f1, fp_fn_ratio)

        results["f1"][x_key] = np.mean(f1_scores)
        results["fp_fn"][x_key] = np.mean(fp_fn_ratios)

        if x_key not in ("SC Max Rsync", "Rsync Max Rsync"):
            performances.append(np.mean(f1_scores))

    plot_path = f"{plot_dir}/{round(w_exc_intra_cluster_ratio, 1)}.png"
    fam_barplot(performances, plot_path)

    queue.put(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_patterns', type=int, default=3,  # 2
                        help="number of input patterns")
    args = parser.parse_args()
    n_patterns = int(args.n_patterns * 20)

    length = 500
    model_class = NetworkLIF
    n_sources = 1
    rate_mean = 50

    # parameters: CONNECTIVITY
    w_ff_val = 3
    pattern_size = 100
    n_neurons = 1000
    w_exc_intra_cluster_ratio = 0.8

    # parameters: EXPERIMENT
    rate_var = 40
    pattern_active_mask = 0.0

    input_size = int(pattern_size * 1.0)
    pattern_active = int(input_size * (1 - pattern_active_mask))

    model_params = params_fixed.copy()

    for n_patterns in (140,):
        x_sc = []
        x_rsync = []
        y = []

        ks_stat_dict = {"sc": [], "rsync": []}
        ks_p_dict = {"sc": [], "rsync": []}
        kl_dict = {"sc": [], "rsync": []}

        fig, ax = plt.subplots(figsize=(20, 7), nrows=2, ncols=5)
        plot_dir = f"plots/abstract/{n_patterns}"
        os.makedirs(plot_dir, exist_ok=True)

        min_rate, max_rate = rate_mean - rate_var, rate_mean + rate_var
        input_rates = np.arange(int(min_rate), int(max_rate), 1)
        n_samples = 160
        iterations = max(1, n_samples // len(input_rates))

        manager = mp.Manager()
        queue = manager.Queue()
        processes = []
        f1_fam = {
            "Ratio": [],
            "Rsync Input": [], "Rsync Max Rsync": [], "Rsync Max SC": [], "Rsync Total": [],
            "SC Input": [], "SC Max Rsync": [], "SC Max SC": [], "SC Total": [],
        }
        fp_fn_fam = {
            "Ratio": [],
            "Rsync Input": [], "Rsync Max Rsync": [], "Rsync Max SC": [], "Rsync Total": [],
            "SC Input": [], "SC Max Rsync": [], "SC Max SC": [], "SC Total": [],
        }
        class_f1 = {
            "Ratio": [],
            "Max pattern Rsync": [], "Max pattern SC": [],
            "Overlap Familiar": [], "Overlap Novel": []
        }

        # Feedforward input connections
        w_ff = build_input_one_to_one(n_neurons=n_neurons, n_inputs=n_neurons) * w_ff_val

        for w_exc_intra_cluster_ratio in np.arange(0.1, 1.1, 0.1):
            p = mp.Process(target=process_ratio, args=(w_exc_intra_cluster_ratio, queue, model_class, model_params,
                                                       n_neurons, w_ff, length, input_rates, iterations,
                                                       n_patterns, pattern_active, pattern_size, n_sources, plot_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        while not queue.empty():
            result = queue.get()
            for key in f1_fam:
                f1_fam[key].append(result["f1"][key])
                fp_fn_fam[key].append(result["fp_fn"][key])
            for key in class_f1:
                class_f1[key].append(result["class_f1"][key])

        # Plot Classification results: F1 and max=pattern proportion
        df = pd.DataFrame(class_f1)
        df = df.sort_values(by=["Ratio"], ascending=True)
        df.to_csv(f"{plot_dir}/class.csv", index=False)

        # Classification performance
        #baseline_class = np.random.choice(np.arange(1, n_patterns + 2), size=1000)

        fig, ax = plt.subplots(figsize=(5, 8), nrows=2, ncols=1)
        ax[0].plot(df["Ratio"], df["Max pattern Rsync"], color="orangered", label="Rsync")
        ax[0].plot(df["Ratio"], df["Max pattern SC"], color="teal", label="Spike count")
        ax[0].set_title("Top pattern = Input pattern", fontsize=20)
        # ax[0].set_xlabel("Intra-pattern connections", fontsize=16)
        ax[0].set_ylabel("F1 score", fontsize=18)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)
        ax[0].set_ylim(0.0, 1.05)
        #ax[0].axhline(y=0.012, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")
        ax[0].legend(fontsize=16)

        ax[1].plot(df["Ratio"], df["Overlap Familiar"], color="cadetblue", label="Familiar")
        ax[1].plot(df["Ratio"], df["Overlap Novel"], color="cadetblue", linestyle="dotted", linewidth=1.75, label="Novel")
        ax[1].set_title("Overlap: Top SC & Input pattern", fontsize=20)
        ax[1].set_xlabel("Intra-pattern connections", fontsize=18)
        ax[1].set_ylabel("Overlap", fontsize=18)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].tick_params(axis='x', labelsize=16)
        ax[1].tick_params(axis='y', labelsize=16)
        ax[1].set_ylim(0.0, 1.05)
        ax[1].axhline(y=0.125, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")
        ax[1].legend(fontsize=16)

        fig.tight_layout()
        fig.savefig(f"{plot_dir}/summary class.png")
        fig.savefig(f"{plot_dir}/summary class.svg")

        # Summary
        fig, ax = plt.subplots(figsize=(5, 8), nrows=2, ncols=1)
        ax[0].plot(df["Ratio"], df["Max pattern Rsync"], color="orangered", label="Rsync")
        ax[0].plot(df["Ratio"], df["Max pattern SC"], color="teal", label="Spike count")
        ax[0].set_title(f"Pattern classification", fontsize=20)
        #ax[0].set_xlabel("Intra-pattern connections", fontsize=18)
        ax[0].set_ylabel("Weighted F1 score", fontsize=18)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)
        ax[0].set_ylim(0.0, 1.05)
        #ax[0].legend(fontsize=16)

        df = pd.DataFrame(f1_fam)
        df = df.sort_values(by=["Ratio"], ascending=True)
        ax[1].plot(df["Ratio"], df["Rsync Max SC"], color="orangered", label="Rsync")
        ax[1].plot(df["Ratio"], df["SC Max SC"], color="teal", label="Spike count")
        ax[1].set_title("Familiarity detection", fontsize=20)
        ax[1].set_xlabel("Intra-pattern connections", fontsize=18)
        ax[1].set_ylabel("F1 score", fontsize=18)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].tick_params(axis='x', labelsize=16)
        ax[1].tick_params(axis='y', labelsize=16)
        ax[1].set_ylim(0, 1.05)
        ax[1].axhline(y=0.33, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")
        ax[1].legend(fontsize=16)

        fig.tight_layout()
        fig.savefig(f"{plot_dir}/summary.png")
        fig.savefig(f"{plot_dir}/summary.svg")

        # Plot Familiarity results: F1 and error analysis
        for label, data in {"F1 score": (f1_fam, (0.0, 1.05)), "Error imbalance": (fp_fn_fam, (-1.05, 1.05))}.items():
            values, lims = data

            df = pd.DataFrame(values)
            df = df.sort_values(by=["Ratio"], ascending=True)
            df.to_csv(f"{plot_dir}/fam {label}.csv", index=False)

            fig, ax = plt.subplots(figsize=(5, 8), nrows=2, ncols=1)
            ax[0].plot(df["Ratio"], df["Rsync Input"], color="orangered", label="Rsync")
            ax[0].plot(df["Ratio"], df["SC Input"], color="teal", label="Spike count")
            ax[0].set_title("Input pattern", fontsize=20)
            #ax[0].set_xlabel("Intra-pattern connections", fontsize=16)
            ax[0].set_ylabel(label, fontsize=18)
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].tick_params(axis='x', labelsize=16)
            ax[0].tick_params(axis='y', labelsize=16)
            ax[0].set_ylim(lims[0], lims[1])
            if label == "F1 score":
                ax[0].axhline(y=0.33, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")
            ax[0].legend(fontsize=16)


            ax[1].plot(df["Ratio"], df["Rsync Max SC"], color="orangered", label="Rsync")
            ax[1].plot(df["Ratio"], df["SC Max SC"], color="teal", label="Spike count")
            ax[1].set_title("Top SC pattern", fontsize=20)
            ax[1].set_xlabel("Intra-pattern connections", fontsize=18)
            ax[1].set_ylabel(label, fontsize=18)
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].tick_params(axis='x', labelsize=16)
            ax[1].tick_params(axis='y', labelsize=16)
            ax[1].set_ylim(lims[0], lims[1])
            if label == "F1 score":
                ax[1].axhline(y=0.33, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")

            fig.tight_layout()
            fig.savefig(f"{plot_dir}/summary {label}.png")
            fig.savefig(f"{plot_dir}/summary {label}.svg")



