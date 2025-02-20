import numpy as np

def measure_rsync(firings):
    def exp_convolve(spike_train):
        tau = 3.0 # ms
        exp_kernel_time_steps = np.arange(0, tau*10, 1)
        decay = np.exp(-exp_kernel_time_steps/tau)
        exp_kernel = decay
        return np.convolve(spike_train, exp_kernel, 'same') # 'valid'

    firings = np.apply_along_axis(exp_convolve, 1, firings)
    meanfield = np.mean(firings, axis=0) # spatial mean across cells, at each time
    variances = np.var(firings, axis=1)  # variance over time of each cell
    rsync = np.var(meanfield) / np.mean(variances)
    if np.isnan(rsync):
        rsync = 0.0
    return rsync

def measure_mean_sc(firings):
    return firings.sum(axis=1).mean() * 2

def measure_max_sc(firings):
    sums = firings.sum(axis=1)
    return firings.sum(axis=1).max() * 2
