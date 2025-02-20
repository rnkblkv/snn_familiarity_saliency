import math
import numpy as np

from sklearn.metrics import f1_score, accuracy_score

from utils import generate_random_patterns, generate_poisson_input, \
                build_pattern_connectivity, build_input_one_to_one,\
                plot_network_activity

class Experiment:
    def __init__(self, metrics=None, length=None):
        """
        Base class for running a simulation experiment
        Args:
            metrics (dict):  keys are metric names, values are metric functions
            length (int): length of simulation in ms
            data (array of tuples OR None): if array of tuples, then stimuli + labels
            data_params (dict or None): if dict, then parameters for continuous data generation with values
        """
        self.metrics = metrics
        self.length = length

    @staticmethod
    def evaluate_mult_metric(res_dict, threshold_dict, y_true):
        y_pred = {}
        for metric in res_dict:
            y_pred[metric] = [1 if r > threshold_dict[metric] else 0 for r in res_dict[metric]]

        best_score = {}

        for metric in y_pred:
            best_score[metric] = f1_score(np.array(y_true), np.array(y_pred[metric]))
        return best_score

    @staticmethod
    def evaluate(res, y_true, threshold=None, metric="rsync"):
        """
        Returns accuracy of a binary classification task.
        Classification procedure:
            1) Find an optimal threshold metric (e.g. rate or synchrony) value
            2) Values over the threshold would correspond to one class, values below - to another one

        Args:
            res (numpy array): array of metric values (e.g. rate or synchrony) for every data sample
            y_true (list or numpy array): ground truth metrics for each data sample (1 if familiar, 0 otherwise)
        """
        if threshold is not None:
            best_threshold = threshold
            y_pred = [1 if r > threshold else 0 for r in res]

            best_score = f1_score(y_true, y_pred)

        else:
            res_mean, res_std = np.mean(res), np.std(res)
            best_threshold, best_score = 0, 0

            if res_mean > 0:
                threshold_min, threshold_max = max(res_mean - res_std*4, np.min(res)), min(res_mean + res_std*4, np.max(res))
                threshold_step = (threshold_max - threshold_min) / 40

                for threshold in np.arange(threshold_min, threshold_max, threshold_step):
                    threshold = round(threshold, 4)
                    y_pred = [1 if r > threshold else 0 for r in res]

                    acc = f1_score(y_true, y_pred)
                    if acc > best_score:
                        best_score, best_threshold = acc, threshold

        return round(best_score, 4), best_threshold
        
class AssociativeFamiliarity(Experiment):
    
    def __init__(self, metrics, length,
                 pattern_size, n_patterns, n_neurons,
                 neuron_measure, energy_measure,
                 input_rate_var=0, input_active_new_var=0, input_active_fam_in_var=0
                 ):
        """
        Class for continual familiarity experiments with plasticity

        Args:
            metrics (dict):  keys are metric names, values are metric functions
            simulation length_test (int): length of test simulation part in ms
            simulation length (int): total length of the simulation in ms
            pattern_params (dict or None): parameters for input patterns generation 
        """
        super().__init__(metrics, length)
        self.n_neurons      = n_neurons
        self.neuron_measure = neuron_measure
        self.energy_measure = energy_measure
        self.n_patterns     = n_patterns
        self.pattern_size   = pattern_size
        self.patterns       = list(generate_random_patterns(
                                        n_neurons=self.n_neurons,
                                        pattern_size=self.pattern_size,
                                        n_patterns=self.n_patterns)
                                 )
        self.input_rate_var           = input_rate_var
        self.input_active_new_var     = input_active_new_var
        self.input_active_fam_in_var  = input_active_fam_in_var

    def run(self, model_class, model_params, thresholds=None, optimize=True, **kwargs):
        """
            model_class (class): a spiking model class
            model_params (dict): parameters for model initialization
        """
        res = {metric: {"score": [], "energy": [], "completedness": []} for metric in self.metrics}
                                   
        w_lat = build_pattern_connectivity(n_neurons=self.n_neurons, patterns=self.patterns, loc=model_params["lat_weight"])
        w_inh = np.zeros(shape=(self.n_neurons, self.n_neurons)) - 1
        np.fill_diagonal(w_inh, 0)
        w_inh *= model_params["inh_weight"]

        # model_params["input_weight"] = 47  # w_lat.max() * 10  # TODO: check whether the constant is better
        w_ext = build_input_one_to_one(n_neurons=self.n_neurons, n_inputs=self.n_neurons) * model_params["input_weight"]

        print("w_lat", w_lat.min(), w_lat.mean(), w_lat.max())
        print("w_ext", w_ext.min(), w_ext.mean(), w_ext.max())
        print("w_inh", w_inh.min(), w_inh.mean(), w_inh.max())

        model = model_class(n_neurons=self.n_neurons, n_inputs=self.n_neurons,
                            w_ext=w_ext, w_lat=w_lat, w_inh=w_inh,
                            **model_params)
                                   
        time_steps = np.arange(0, self.length + model_params["dt"], model_params["dt"])
        input_size = self.pattern_size
                                      
        all_input = []
        y_true = []
        input_rate_loc = 75

        #selected_patterns = np.random.choice(np.arange(self.n_patterns), 30, replace=True)
        min_rate = max(1, int(input_rate_loc * (1 - self.input_rate_var)))
        max_rate = int(input_rate_loc * (1 + self.input_rate_var)) + 1

        min_input_size_new = max(1, int(input_size * (1 - self.input_active_new_var)))
        max_input_size_new = int(input_size * (1 + self.input_active_new_var)) + 1
        min_input_size_fam = max(1, int(input_size * (1 - self.input_active_fam_in_var)))
        max_input_size_fam = input_size + 1
        pattern_active = int(input_size * (1 - self.input_active_fam_in_var))

        for i, input_rate in enumerate(range(min_rate, max_rate, 1)):

            verbose = False
            if i % 10 == 0:
                verbose = True

            for j in range(3):
                pattern_idx = np.random.choice(np.arange(self.n_patterns))
                for fam in (False, True):

                    if fam is False:
                        random_input_size = np.random.choice(np.arange(min_input_size_new, max_input_size_new))
                        pattern_cur = sorted(np.random.choice(np.arange(self.n_neurons),
                                                              random_input_size,
                                                              replace=False))
                    else:
                        # pattern_active = np.random.choice(np.arange(min_input_size_fam, max_input_size_fam))
                        pattern_cur = sorted(np.random.choice(self.patterns[pattern_idx],
                                                                 pattern_active,
                                                                 replace=False))
                    # print("PATTERN", fam, "var", self.input_active_new_var, "len pattern cur", len(pattern_cur))
                    y_true.append(int(fam))

                    # poisson encoding
                    poisson_input_rates = np.zeros(self.n_neurons)
                    poisson_input_rates[pattern_cur] = input_rate
                    poisson_input = generate_poisson_input(time_steps, rates=poisson_input_rates)
                    all_input.append(poisson_input)

                    model_kwargs = {'sample': str(i), 'proc_name': ''}
                    if 'proc_name' in kwargs:
                        model_kwargs['proc_name'] = kwargs['proc_name']

                    # Decide stimulus familiarity
                    voltage, spikes = model.simulate(length=self.length,
                                                     external_input=poisson_input
                                                     )
                    res_print = {}
                    neuron_measure = np.arange(spikes.shape[0])
                    pattern_max_idx = {metric: 0 for metric in self.metrics}
                    pattern_measure_max = {metric: 0 for metric in self.metrics}

                    for metric in self.metrics:
                        for pattern_check_idx, pattern_check in enumerate(self.patterns):

                            if not (fam is True and pattern_check_idx == pattern_idx):
                                metric_func = self.metrics[metric]['func']
                                metric_func_kwargs = self.metrics[metric]['kwargs']
                                pattern_measure = metric_func(spikes[pattern_check], **metric_func_kwargs)

                                if pattern_measure >= pattern_measure_max[metric]:
                                    pattern_max_idx[metric] = int(pattern_check_idx)
                                    pattern_measure_max[metric] = float(pattern_measure)

                    neuron_orig = self.patterns[pattern_idx].copy() if fam is True else pattern_cur.copy()
                    if self.neuron_measure == "orig":
                        neuron_measure = neuron_orig.copy()

                    for metric in self.metrics:
                        metric_func = self.metrics[metric]['func']
                        metric_func_kwargs = self.metrics[metric]['kwargs']

                        score = metric_func(spikes[neuron_measure], **metric_func_kwargs)

                        # if no neurons fired:s
                        if np.isnan(score):
                            score = 0
                        """
                        # if there is an epileptic-like activity, consider pattern not familiar
                        elif fam is True and pattern_idx != pattern_max and self.metrics["rsync"]["func"](spikes) >= 0.1:
                            correction = 0
                        
                        # if the pattern is not completed, consider it non-familiar
                        elif fam is True and self.input_active_fam_in_var > 0:
                            n_completed = sum(spikes[self.patterns[pattern_idx]].sum(1) > spikes.sum(1).mean()) - len(pattern_cur)
                            n_to_complete = input_size - len(pattern_cur)
                            correction = n_completed / n_to_complete
                        """
                        pattern_max = self.patterns[pattern_max_idx[self.energy_measure]]
                        e1 = metric_func(spikes[neuron_orig], **metric_func_kwargs)  # if fam, then full pattern

                        if fam is True and self.input_active_fam_in_var > 0:  # only pattern neurons not receiving input
                            non_active = [n for n in self.patterns[pattern_idx] if n not in pattern_cur]
                            e1 = metric_func(spikes[non_active], **metric_func_kwargs)

                        e2 = metric_func(spikes[pattern_cur], **metric_func_kwargs)  # neurons receiving input
                        e3 = metric_func(spikes, **metric_func_kwargs)  # all neurons
                        energy = min(1, math.e ** ((e3 - e1) / (e3 + 0.0001)))
                        energy = round(energy, 4)

                        completedness = math.e ** ((e1 - e2) / (e1 + 0.0001))
                        completedness = round(completedness, 4)

                        res[metric]["score"].append(score)
                        res_print[metric] = round(score, 3)
                        res[metric]["energy"].append(energy)

                        if fam is True:
                            res[metric]["completedness"].append(completedness)

                    if verbose:
                        print(f'{model_kwargs["proc_name"]} {input_rate}_{j} {fam} {len(pattern_cur)} {res_print} energy {energy}')
                        #plot_network_activity(voltage, spikes)
        out_score = {}
        for metric in res:
            out_score[metric] = {}
            threshold = None
            if thresholds is not None:
                threshold = thresholds.get(metric, None)
            score, res_threshold = Experiment.evaluate(np.array(res[metric]["score"]), np.array(y_true),
                                                       threshold, metric)
            out_score[metric]['energy'] = np.mean(res[metric]["energy"])
            out_score[metric]["completedness"] = np.mean(res[metric]["completedness"])
            if optimize is True:
                out_score[metric]['score'] = score - max(np.mean(res[self.energy_measure]['energy']), 1 - np.mean(res[self.energy_measure]['completedness']))  # TODO: check second part
            out_score[metric]['threshold'] = res_threshold

        if optimize is False:
            thresholds = {metric: out_score[metric]['threshold'] for metric in out_score}
            mult_metric_scores = Experiment.evaluate_mult_metric(res, thresholds, np.array(y_true))

            for metric in mult_metric_scores:
                out_score[metric] = {"score": mult_metric_scores[metric]}

        return model, out_score
