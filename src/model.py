import numpy as np

class NetworkLIF:
    def __init__(self, dt=0.5, V_thr=-55, V_noise=0.0, E_L=-65, tau_m=10.0, g_L=10.0, t_refr=2.0,
                 n_neurons=10, n_inputs=10, w_ff=None, w_exc=None, w_inh=None,
                 **kwargs):
        '''
        Simulate the activity of a LIF neuron.

        Args:
            V_thr     -- the threshold for neuron firing
            V_noise   -- random noise added to the signal
            dt        -- time step for the simulation
            n_neurons -- number of neurons in the network
            n_inputs  -- number of input sources to the neurons
            w_ff     -- feedforward connectivity matrix from the input layer to LIF layer
            w_exc    -- lateral connectivity matrix within the LIF layer
        Constant params:
            g_L   -- leak of the conductance (inverse membrane resistance)
            tau_m -- membrane time constant: how quickly membrane voltage returns to the resting state
            E_L   -- resting potential (equal to voltage reset after spike)
        '''
        self.dt    = dt
        self.E_L   = E_L
        self.tau_m = tau_m
        self.g_L   = g_L

        self.V_thr   = V_thr
        self.V_noise = V_noise
        self.t_refr  = t_refr

        self.n_inputs  = n_inputs
        self.n_neurons = n_neurons

        self.w_ff = w_ff
        if not isinstance(w_ff, np.ndarray):
            self.w_ff = np.zeros(shape=((n_neurons, n_inputs)))
            np.fill_diagonal(self.w_ff, 1)
        elif w_ff.shape  != (n_neurons, n_inputs):
            raise Exception(f"The shape of the input connectivity matrix should be ({n_neurons}, {n_inputs}), but it is {w_ext.shape}")

        self.w_exc = w_exc
        if not isinstance(w_exc, np.ndarray):
            self.w_exc = np.zeros(shape=((n_neurons, n_neurons)))
        elif w_exc.shape  != (n_neurons, n_neurons):
            raise Exception(f"The shape of the lateral connectivity matrix should be ({n_neurons}, {n_neurons}), but it is {w_lat.shape}")

        self.w_inh = w_inh
        if not isinstance(w_inh, np.ndarray):
            self.w_inh = np.zeros(shape=((n_neurons, n_neurons)))
        elif w_inh.shape  != (n_neurons, n_neurons):
            raise Exception(f"The shape of the inhibitory connectivity matrix should be ({n_neurons}, {n_neurons}), but it is {w_inh.shape}")
            
    def _time_step(self, V, I, refr):
        
        # Voltage update
        noise   = np.random.rand() * self.V_noise
        # insert your formula for the membrane potential (voltage) update
        #dV = noise + (-(V - self.E_L) + I / self.g_L) / self.tau_m
        dV = noise + (-(V - self.E_L) / self.tau_m + I)
        # integrate the above update
        V += dV * self.dt

        # refractory
        V[refr > 0] = self.E_L
        refr[refr > 0] -= 1

        fired = V > self.V_thr
        # what happens to the neurons whose membrane potential exceeded a threshold?
        V[fired]    = self.E_L
        refr[fired] = self.t_refr / self.dt
        return V, fired, refr
    
    def simulate(self, length, external_input=None, input_scale=None):
        '''
        Args:
            external_input -- input to the neuron
            length         -- simulation length [ms]
        '''

        time_steps    = np.arange(0, length + self.dt, self.dt)      # simulation time steps [ms]
        voltage       = np.zeros((self.n_neurons, len(time_steps)))  # array for saving voltage history
        voltage[:, 0] = self.E_L                                     # set initial voltage to resting potential
        spikes        = np.zeros((self.n_neurons, len(time_steps)))  # initialize spike output
        refr          = np.zeros((self.n_neurons,))

        # simulation
        for t in range(1, len(time_steps)):
            # calculate input to the model: a sum of the spiking inputs weighted by corresponding connections
            ff_input = np.dot(self.w_ff, external_input[:, t])
            exc_input = np.dot(self.w_exc, spikes[:, t-1])
            inh_input = np.dot(self.w_inh, spikes[:, t-1]) 
            total_input = ff_input + exc_input + inh_input

            #if total_input.sum() > 0:
            #    print(t, total_input.nonzero()[0], external_input[:, t].nonzero()[0])

            # update voltage and record spikes
            voltage[:, t], spikes[:, t], refr = self._time_step(voltage[:, t-1], total_input, refr.copy())
            
        return voltage, spikes