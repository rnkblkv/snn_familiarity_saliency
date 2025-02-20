import numpy as np
from scipy import stats


# ALREADY IN UTILS
def truncated_normal(size, mean, std, lower, upper):
    """Generate values from a truncated normal distribution."""
    a, b = (lower - mean) / std, (upper - mean) / std
    return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def generate_neuron_indices(grid_size, orientations):
    """
    Generate 3D and 1D representations of neurons filled with orientation values.

    Parameters:
    - grid_size: int, size of one dimension of the visual field grid (total neurons = grid_size^2 * len(orientations))
    - orientations: list, possible orientation preferences

    Returns:
    - neurons_3d: np.array, shape (num_orientations, grid_size, grid_size), filled with orientation values
    - neurons_1d: np.array, flattened version of neurons_3d
    """
    num_orientations = len(orientations)
    neurons_3d = np.array(orientations).reshape(num_orientations, 1, 1) * np.ones(
        (num_orientations, grid_size, grid_size))
    neurons_1d = neurons_3d.flatten()

    return neurons_3d, neurons_1d


def index_3d_to_1d(orientation_idx, x, y, grid_size):
    """Convert 3D neuron index to 1D index."""
    return orientation_idx * grid_size * grid_size + x * grid_size + y


def index_1d_to_3d(index, grid_size, num_orientations):
    """Convert 1D neuron index to 3D index."""
    orientation_idx = index // (grid_size * grid_size)
    remaining = index % (grid_size * grid_size)
    x = remaining // grid_size
    y = remaining % grid_size
    return orientation_idx, x, y


def get_orientation_indices(grid_size, orientations):
    """
    Generate indices of neurons for each orientation in both 1D and 3D.

    Parameters:
    - grid_size: int, size of one dimension of the visual field grid
    - orientations: list, possible orientation preferences

    Returns:
    - orientation_indices_3d: dict, mapping each orientation to its indices in the 3D array
    - orientation_indices_1d: dict, mapping each orientation to its indices in the 1D array
    """
    num_orientations = len(orientations)
    orientation_indices_3d = {}
    orientation_indices_1d = {}

    for i, ori in enumerate(orientations):
        indices_3d = np.array([[i, x, y] for x in range(grid_size) for y in range(grid_size)])
        indices_1d = np.array([index_3d_to_1d(i, x, y, grid_size) for x in range(grid_size) for y in range(grid_size)])

        orientation_indices_3d[ori] = indices_3d
        orientation_indices_1d[ori] = indices_1d

    return orientation_indices_3d, orientation_indices_1d


def compute_orientation_difference_matrix(neurons_1d, orientations):
    """
    Compute an n_neurons x n_neurons matrix storing orientation differences.

    Parameters:
    - neurons_1d: np.array, flattened array of neuron orientations
    - orientations: list, possible orientation preferences

    Returns:
    - orientation_diff_matrix: np.array, shape (n_neurons, n_neurons)
    """
    n_neurons = len(neurons_1d)

    def orientation_difference(o1, o2):
        return np.minimum(np.abs(o1 - o2), 180 - np.abs(o1 - o2))

    # Compute orientation differences
    orientation_diff_matrix = orientation_difference(neurons_1d[:, None], neurons_1d[None, :])

    return orientation_diff_matrix


def truncated_lognormal(mean, sigma, lower, upper, size):
    """Generate samples from a truncated lognormal distribution."""

    # Compute the corresponding normal mean μ
    mu = np.log(mean) - 0.5 * sigma ** 2

    # Compute truncation limits in normal space
    a, b = (np.log(lower) - mu) / sigma, (np.log(upper) - mu) / sigma

    # Generate samples from truncated normal, then exponentiate
    truncated_samples = stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)
    lognormal_samples = np.exp(truncated_samples)

    return lognormal_samples


def generate_connections_exc(orientation_diff_matrix, distance_matrix, connection_fraction=0.1):
    # Given probability distributions

    orientation_differences_probs = {0: 0.5, 22.5: 0.2, 45: 0.09, 67.5: 0.06, 90: 0.15}
    distance_probs = {0: 0.24, 1: 0.34, 2: 0.21, 3: 0.1, 4: 0.06, 5: 0.006, 6: 0.006, 7: 0.006, 8: 0.006, 9: 0.006}

    n_neurons = orientation_diff_matrix.shape[0]  # Number of neurons
    n_connections = int(connection_fraction * n_neurons)  # Outgoing connections per neuron
    print(n_connections, "E")
    # Initialize connection matrix with zeros
    connection_matrix = np.zeros((n_neurons, n_neurons), dtype=int)

    # Convert probability dictionaries to arrays for efficient lookup
    orientation_prob_array = np.vectorize(orientation_differences_probs.get)(orientation_diff_matrix)
    orientation_prob_array = orientation_prob_array / orientation_prob_array.sum(axis=1, keepdims=True)

    distance_prob_array = np.vectorize(distance_probs.get)(distance_matrix)
    distance_prob_array = distance_prob_array / distance_prob_array.sum(axis=1, keepdims=True)

    # Compute combined probability by multiplying orientation and distance probabilities
    combined_prob = orientation_prob_array * distance_prob_array

    # Normalize probabilities row-wise to form a valid probability distribution
    combined_prob = combined_prob / combined_prob.sum(axis=1, keepdims=True)

    # Generate connections for each neuron
    for i in range(n_neurons):
        if np.isnan(combined_prob[i]).any():
            continue  # Skip rows with NaNs (if any)

        # Select target neurons based on the computed probabilities
        target_neurons = np.random.choice(n_neurons, size=n_connections, replace=False, p=combined_prob[i])

        # Mark selected connections in the matrix
        connection_matrix[target_neurons, i] = 1  # Incoming connections to neuron i

    return connection_matrix


def generate_w_exc(orientation_diff_matrix, distance_matrix, mean_weight, sigma_weight, connection_fraction):
    """
    Generate a weight matrix for connections based on a log-normal distribution.

    Parameters:
    - connectivity_matrix: np.array, binary matrix indicating connections
    - mean_weight: float, mean of the log-normal distribution
    - max_weight: float, maximum possible weight
    - sigma: float, standard deviation of the log-normal distribution

    Returns:
    - weight_matrix: np.array, same shape as connectivity_matrix with assigned weights
    """
    connectivity_matrix = generate_connections_exc(orientation_diff_matrix, distance_matrix, connection_fraction)
    n_neurons = connectivity_matrix.shape[0]

    raw_weights = truncated_lognormal(mean=mean_weight, sigma=sigma_weight, lower=0.1, upper=5,
                                      size=(n_neurons, n_neurons))

    # Apply weights only where connections exist
    weight_matrix = connectivity_matrix * raw_weights
    np.fill_diagonal(weight_matrix, 0)

    return weight_matrix


def compute_chebyshev_distance(neuron_positions):
    """
    Compute an n_neurons x n_neurons matrix storing Chebyshev distances between neurons.

    Parameters:
    - neuron_positions: np.array, shape (n_neurons, 2), containing (x, y) positions of neurons

    Returns:
    - distance_matrix: np.array, shape (n_neurons, n_neurons)
    """
    n_neurons = neuron_positions.shape[0]

    # Compute pairwise Chebyshev distances
    distance_matrix = np.max(np.abs(neuron_positions[:, None, :] - neuron_positions[None, :, :]), axis=-1)

    return distance_matrix


def generate_connections_som(orientation_diff_matrix, distance_matrix, connection_fraction=0.1, max_distance=3):
    """
    Generates a connection matrix for SOM interneurons where connection probability depends on:
    - Distance between neurons
    - Orientation difference between neurons
    """
    # Given probability distributions
    distance_probs = {0: 0.3, 1: 0.2, 2: 0.12, 3: 0.07}  # Probabilities for distance
    orientation_diff_probs = {0: 0.5, 22.5: 0.2, 45: 0.09, 67.5: 0.06, 90: 0.15}  # Probabilities for orientation difference

    n_neurons = distance_matrix.shape[0]  # Number of neurons

    # Create a mask for valid connections (distance ≤ max_distance)
    valid_mask = distance_matrix <= max_distance

    # Initialize connection matrix with zeros
    connection_matrix = np.zeros((n_neurons, n_neurons), dtype=int)

    # Apply probability mapping
    orientation_prob_array = np.vectorize(orientation_diff_probs.get)(orientation_diff_matrix)
    orientation_prob_array = orientation_prob_array / orientation_prob_array.sum(axis=1, keepdims=True)

    distance_prob_array = np.vectorize(lambda d: distance_probs.get(d, 0))(distance_matrix)
    distance_prob_array = distance_prob_array / distance_prob_array.sum(axis=1, keepdims=True)

    # Compute final joint probability by multiplying both distributions
    combined_prob_array = distance_prob_array * orientation_prob_array

    # Zero out probabilities for invalid distances
    combined_prob_array[~valid_mask] = 0

    # Generate connections for each neuron
    for i in range(n_neurons):
        # Get valid targets
        valid_targets = np.where(valid_mask[i])[0]  # Neurons with valid distances

        if len(valid_targets) == 0:
            continue  # No valid targets for this neuron

        # Compute fraction based on the number of valid targets
        n_connections = int(connection_fraction * len(valid_targets))
        if n_connections == 0:
            continue  # No connections if the fraction rounds to zero

        # Normalize probabilities for valid targets
        valid_probs = combined_prob_array[i, valid_targets]
        valid_probs /= valid_probs.sum()  # Normalize to sum to 1

        # Select target neurons based on computed probabilities
        target_neurons = np.random.choice(valid_targets, size=n_connections, replace=False, p=valid_probs)

        # Mark selected connections in the matrix
        connection_matrix[target_neurons, i] = 1  # Incoming connections to neuron i

    return connection_matrix


def generate_w_som(orientation_diff_matrix, distance_matrix,
                   connection_fraction, mean_weight, sigma_weight):
    connectivity_matrix = generate_connections_som(orientation_diff_matrix, distance_matrix, connection_fraction)
    n_neurons = connectivity_matrix.shape[0]

    raw_weights = truncated_lognormal(mean=mean_weight, sigma=sigma_weight, lower=0.1, upper=5,
                                      size=(n_neurons, n_neurons))

    # Apply weights only where connections exist
    weight_matrix = connectivity_matrix * raw_weights
    np.fill_diagonal(weight_matrix, 0)

    return weight_matrix * (-1)


def generate_connections_pv(distance_matrix, connection_fraction=0.1, max_distance=3):
    # Given probability distribution
    distance_probs = {0: 0.3, 1: 0.2, 2: 0.12, 3: 0.07}

    n_neurons = distance_matrix.shape[0]  # Number of neurons

    # Create a mask for valid connections (distance ≤ max_distance)
    valid_mask = distance_matrix <= max_distance

    # Initialize connection matrix with zeros
    connection_matrix = np.zeros((n_neurons, n_neurons), dtype=int)

    # Apply probability mapping to distance matrix
    distance_prob_array = np.vectorize(lambda d: distance_probs.get(d, 0))(distance_matrix)

    # Zero out probabilities for invalid distances
    distance_prob_array[~valid_mask] = 0

    all_con = []
    # Generate connections for each neuron
    for i in range(n_neurons):
        # Get valid targets
        valid_targets = np.where(valid_mask[i])[0]  # Neurons with valid distances

        if len(valid_targets) == 0:
            continue  # No valid targets for this neuron

        # Compute fraction based on the number of valid targets
        n_connections = int(connection_fraction * len(valid_targets))
        all_con.append(n_connections)
        if n_connections == 0:
            continue  # No connections if the fraction rounds to zero

        # Normalize probabilities for valid targets
        valid_probs = distance_prob_array[i, valid_targets]
        valid_probs /= valid_probs.sum()  # Normalize to sum to 1

        # Select target neurons based on computed probabilities
        target_neurons = np.random.choice(valid_targets, size=n_connections, replace=False, p=valid_probs)

        # Mark selected connections in the matrix
        connection_matrix[target_neurons, i] = 1  # Incoming connections to neuron i
    print(np.mean(all_con), "PV")
    return connection_matrix


def generate_w_pv(distance_matrix, connection_fraction,
                  mean_weight, sigma_weight):
    connectivity_matrix = generate_connections_pv(distance_matrix, connection_fraction)
    n_neurons = connectivity_matrix.shape[0]

    raw_weights = truncated_lognormal(mean=mean_weight, sigma=sigma_weight, lower=0.1, upper=5,
                                      size=(n_neurons, n_neurons))

    # Apply weights only where connections exist
    weight_matrix = connectivity_matrix * raw_weights
    np.fill_diagonal(weight_matrix, 0)

    return weight_matrix * (-1)


def build_connectivity(grid_size):
    orientations = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]

    neuron_positions = np.array([(x, y) for x in range(grid_size) for y in range(grid_size)] * len(orientations))
    distance_matrix = compute_chebyshev_distance(neuron_positions)

    neurons_3d, neurons_1d = generate_neuron_indices(grid_size, orientations)
    orientation_diff_matrix = compute_orientation_difference_matrix(neurons_1d, orientations)

    w_som_fraction = 0.05
    w_som_mean = 0.4
    w_som_sigma = 0.8
    w_som = generate_w_som(orientation_diff_matrix, distance_matrix,
                           w_som_fraction, w_som_mean, w_som_sigma)

    w_pv_fraction = 0.2
    w_pv_mean = 1.0
    w_pv_sigma = 0.5
    w_pv = generate_w_pv(distance_matrix,
                         w_pv_fraction, w_pv_mean, w_pv_sigma)

    w_exc_fraction = 0.1
    w_exc_mean = 0.4
    w_exc_sigma = 0.8
    w_exc = generate_w_exc(orientation_diff_matrix=orientation_diff_matrix, distance_matrix=distance_matrix,
                           mean_weight=w_exc_mean, sigma_weight=w_exc_sigma,
                           connection_fraction=w_exc_fraction)

    return w_exc, w_som, w_pv