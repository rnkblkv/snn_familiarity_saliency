import logging
import matplotlib.pyplot as plt
import numpy as np
import itertools

from scipy.stats import truncnorm

def create_logger(logger_name):
    """
    Gets or creates a logger
    """
    logger = logging.getLogger(__name__)
    # set log level
    logger.setLevel(logging.INFO)

    # define file handler and set formatter
    file_handler = logging.FileHandler(logger_name)
    formatter    = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # add file handler to logger
    logger.addHandler(file_handler)
    return logger

def exp_convolve(spike_train):
    tau = 3.0  # ms
    exp_kernel_time_steps = np.arange(0, tau * 10, 1)
    decay = np.exp(-exp_kernel_time_steps / tau)
    exp_kernel = decay
    return np.convolve(spike_train, exp_kernel, 'same')  # 'valid'

def plot_neuron_activity(time_steps, voltage=None, spikes=None, external_input=None, lateral_input=None, title=""):
    '''
    Plot voltage and input for one neuron over time
    '''
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)

    # plot spikes
    spike_ids = spikes.nonzero()[0] - 1
    voltage[spike_ids] = 50

    # plot lines
    if external_input is not None:
        line_external = ax.plot(time_steps, np.clip(external_input, 0, 60), label="External input", color="coral")[0]
    if lateral_input is not None:
        line_lateral  = ax.plot(time_steps, np.clip(lateral_input, 0, 50), label="Lateral input", color="firebrick")[0]
    if voltage is not None:
        line_voltage  = ax.plot(time_steps, voltage, label="Membrane potential", color="lightseagreen")[0]

    ax.set_ylim(-80, 90)
    ax.legend(loc="upper right", ncols=2)
    ax.set_xlabel("Membrane potential [mV] / Input current")
    ax.set_ylabel("Time [ms]")
    plt.show()

def plot_network_activity(voltage, spikes):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    h, w = voltage.shape
    
    ax[0].imshow(voltage)
    ax[0].set_title("Voltage")
    ax[0].set_xlabel("Neurons")
    ax[0].set_ylabel("Time [ms]")
    ax[0].set_aspect(w / h)

    ax[1].imshow(spikes)
    ax[1].imshow(spikes)
    ax[1].set_title("Firing activity")
    ax[1].set_ylabel("Neurons")
    ax[1].set_xlabel("Time [ms]")
    ax[1].set_aspect(w / h)
    
    fig.tight_layout()
    plt.show()

def plot_spike_counts(parameters, spike_count_values, title=""):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(title)

    ax.plot(parameters, spike_count_values)
    plt.show()
    
def build_input_one_to_one(n_neurons, n_inputs):
    w = np.zeros(shape=(n_neurons, n_inputs))
    np.fill_diagonal(w, 1)
    return w

def build_input_equal(n_neurons, n_inputs):
    w = np.ones(shape=(n_neurons, n_inputs))
    return w

def build_input_random(n_neurons, n_inputs):
    w = np.random.normal(size=(n_neurons, n_inputs), loc=0.5, scale=0.25)
    return w

def build_input_half(n_neurons, n_inputs):
    w = np.random.normal(size=(n_neurons, n_inputs), loc=0.5, scale=0.25)
    half_neurons = n_neurons // 2
    half_inputs  = n_inputs // 2
    
    w[:half_neurons, half_inputs:] = 0
    w[half_neurons:, :half_inputs] = 0
    return w


def generate_synchronous_input(time_steps, rate, n_neurons, ids):
    sync_inputs = np.zeros(shape=(n_neurons, time_steps))
    input_times = np.arange(0, time_steps, time_steps / rate)[1:].astype(np.int16)
    for i in ids:
        sync_inputs[i, input_times] = 1
    return sync_inputs

def generate_random_patterns(n_neurons, pattern_size, n_patterns):
    patterns = []

    while len(patterns) < n_patterns:
        pattern = np.random.choice(np.arange(n_neurons), pattern_size, False)
        pattern.sort()
        pattern = list(pattern)
        if pattern not in patterns:
            patterns.append(pattern)
    return patterns

def generate_lgn_inputs(n_neurons, n_sources, pattern, input_rate, length, dt=1.0):
    """
    Generate Poisson spike inputs for a population of neurons with two different firing rates.

    Args:
        n_neurons (int): Total number of neurons in the network.
        n_sources (int): Number of Poisson sources per neuron.
        rate_A (float): Firing rate (Hz) for neurons in group A.
        rate_B (float): Firing rate (Hz) for neurons in group B.
        indices_A (list): Indices of neurons in group A (others will be in group B).
        duration_ms (int): Duration of the simulation in ms.
        dt_ms (int): Time step size in ms (default: 1 ms).

    Returns:
        np.ndarray: Binary spike train array of shape (n_neurons, duration_ms).
    """
    # Initialize spike train array (binary spikes for each neuron at each time step)
    spike_trains = np.zeros((n_neurons, length), dtype=int)

    # Define firing rates for each neuron
    rates = np.full(n_neurons, 0)
    rates[pattern] = input_rate

    # Generate Poisson inputs for each neuron
    for neuron in range(n_neurons):
        for _ in range(n_sources):  # Each neuron receives input from n_sources Poisson sources
            spike_times = np.random.rand(length) < (rates[neuron] * dt / 1000.0)
            spike_trains[neuron, :] += spike_times.astype(int)

    # Convert to binary spikes (clip values > 1)
    spike_trains = np.clip(spike_trains, 0, 1)

    return spike_trains

    
