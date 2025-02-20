import itertools
import os

from measure import measure_rsync, measure_mean_sc

data_dir = "data"
log_dir = os.path.join(data_dir, "logs")
plot_dir = os.path.join(data_dir, "plots")
param_dir = os.path.join(data_dir, "params")

params_fixed  = {
                "dt": 1.0,
                "V_thr": -55,
                "E_L": -65,
                "V_noise": 1.5,
                "t_refr": 2.0,
                "tau_m": 10.0,
                "g_L": 10.0,
                "w_ff_val": 5
                }

params_change = {
                "w_exc_mean": {"min": 0.35, "max": 1.0, "step": 0.01},
                "w_som_mean": {"min": 0.0, "max": 5.0, "step": 0.1},
                "w_pv_mean": {"min": 0.0, "max": 5.0, "step": 0.1}
                }

metric_names_test = ["sc", "rsync"]
n_patterns_list = [10, 20, 30, 40, 50]

input_rate_var = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
input_active_new_var = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

input_active_fam_in_var = [0.0, 0.1, 0.2, 0.3, 0.4]

metrics = {"sc": {"func": measure_mean_sc, "kwargs": {}},
           "rsync": {"func": measure_rsync, "kwargs": {}}
           }
#optimization_iter_list = list(range(10))
combinations = list(itertools.product(metric_names_test, metric_names_test,
                                      n_patterns_list, input_active_fam_in_var))
