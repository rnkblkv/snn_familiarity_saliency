import sys
import os
import numpy as np

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.network_v1 import truncated_normal, truncated_lognormal
from src.utils import generate_random_patterns

def generate_w_exc(n_neurons, patterns, p_connect, intra_cluster_ratio, mean_weight, sd_weight):
    """
    Generates the EXC connectivity matrix.

    - Each neuron connects to 10% of all neurons.
    - 80% of connections are within the same cluster.
    - 20% of connections are to other clusters.
    - Weights follow a log-normal distribution.

    Parameters:
    - n (int): Total number of neurons.
    - patterns (list of lists): Each sublist contains indices of neurons in a cluster.
    - p_connect (float): Probability of connection (default 10% for EXC neurons).
    - intra_cluster_ratio (float): Fraction of connections within the same cluster (default 80%).
    - lognorm_mean (float): Mean for log-normal weight distribution.
    - lognorm_std (float): Standard deviation for log-normal weight distribution.

    Returns:
    - exc_matrix (np.ndarray): n × n connectivity matrix for excitatory neurons.
    """
    w = np.zeros((n_neurons, n_neurons))  # Initialize connectivity matrix

    for pattern in patterns:
        other_neurons = list(set(range(n_neurons)) - set(pattern))  # Neurons outside the current pattern

        for neuron in pattern:
            n_selected_total = max(1, int(n_neurons * p_connect))
            n_selected_intra = max(1, int(n_selected_total * intra_cluster_ratio))
            n_selected_inter = max(1, n_selected_total - n_selected_intra)

            # Select intra-cluster connections (from the same cluster)
            selected_intra = np.random.choice(pattern, size=n_selected_intra, replace=False)

            # Select inter-cluster connections (from different clusters)
            selected_inter = np.random.choice(other_neurons, size=n_selected_inter, replace=False)

            # Assign log-normal weights
            weights_intra = truncated_lognormal(mean=mean_weight, sigma=sd_weight, size=n_selected_intra,
                                                          lower=0.1, upper=5.0)
            weights_inter = truncated_lognormal(mean=mean_weight, sigma=sd_weight, size=n_selected_inter,
                                                          lower=0.1, upper=5.0)

            # Update the matrix
            w[neuron, selected_intra] = weights_intra
            w[neuron, selected_inter] = weights_inter

    np.fill_diagonal(w, 0)
    return w


def generate_w_pv(n_neurons, p_connect, mean_weight, sd_weight):
    random_values = np.random.rand(n_neurons, n_neurons)
    w = (random_values < p_connect).astype(int)  # 6%

    raw_weights = truncated_lognormal(mean=mean_weight, sigma=sd_weight, size=(n_neurons, n_neurons),
                                      lower=0.1, upper=10.0)

    w = w * raw_weights
    #np.fill_diagonal(w, 0)

    return w * (-1)

def generate_w_som(n_neurons, patterns, p_connect,
                   mean_weight, sd_weight):

    w = np.zeros((n_neurons, n_neurons))  # Initialize connectivity matrix

    for pattern in patterns:
        other_neurons = list(set(range(n_neurons)) - set(pattern))  # Neurons outside the current pattern

        for neuron in pattern:
            # Select a subset of 8% neurons from other patterns
            n_selected = max(1, int(len(other_neurons) * p_connect))  # Ensure at least 1 connection
            selected_targets = np.random.choice(other_neurons, size=n_selected, replace=False)

            # Assign log-normal weights
            weights = truncated_lognormal(mean=mean_weight, sigma=sd_weight, size=n_selected,
                                          lower=0.1, upper=10.0)
            w[neuron, selected_targets] = weights  # Set weights in the matrix

    return w * (-1)

def build_connectivity(n_neurons=1000, n_patterns=10, pattern_size=100, w_exc_intra_cluster_ratio=0.9):
    patterns = generate_random_patterns(n_neurons, pattern_size, n_patterns)

    w_pv_mean = 1
    w_pv_sd = 0.4
    w_pv_p = 0.06
    w_pv = generate_w_pv(n_neurons=n_neurons, p_connect=w_pv_p,
                         mean_weight=w_pv_mean, sd_weight=w_pv_sd
                         )
    w_som_p = 0.11
    w_som_mean_weight = 1.0
    w_som_sd_weight = 0.5
    w_som = generate_w_som(n_neurons=n_neurons, patterns=patterns, p_connect=w_som_p,
                           mean_weight=w_som_mean_weight, sd_weight=w_som_sd_weight
                           )
    w_exc_mean = 0.4
    w_exc_sd = 0.8
    w_exc_p = 0.1
    w_exc = generate_w_exc(n_neurons=n_neurons, patterns=patterns, p_connect=w_exc_p,
                           intra_cluster_ratio=w_exc_intra_cluster_ratio,
                           mean_weight=w_exc_mean, sd_weight=w_exc_sd)

    return patterns, w_exc, w_som, w_pv