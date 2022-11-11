import math

import torch
import torch.nn as nn


"""
Parameters for SNN
"""

ENCODER_VTH = 1.0

NEURON_VTH = 0.5
NEURON_CDECAY = 0.5
SPIKE_PSEUDO_GRAD_WINDOW = 0.5


class PseudoSpikeDeterministic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_VTH).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class PseudoSpikeRandom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.bernoulli(input).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class PseudoSpikeAtan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(ENCODER_VTH).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input / (1 + (math.pi * (input - ENCODER_VTH)).pow_(2))


class PseudoSpikeRect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()


class PopSpikeEncoderDeterministic(nn.Module):
    """ Learnable Population Coding Spike Encoder with Deterministic Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoSpikeDeterministic.apply
        
        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        obs = obs.view(-1, self.obs_dim, 1)
        
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        
        # Generate Deterministic Spike Trains
        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt)
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_VTH
        
        return pop_spikes


class PopSpikeEncoderRandom(PopSpikeEncoderDeterministic):
    """ Learnable Population Coding Spike Encoder with Random Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)
        self.pseudo_spike = PseudoSpikeRandom.apply

    def forward(self, obs, batch_size):
        obs = obs.view(-1, self.obs_dim, 1)
        
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        
        # Generate Random Spike Trains
        for step in range(self.spike_ts):
            pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
        
        return pop_spikes


class PopSpikeEncoderAtan(nn.Module):
    """ Learnable Population Coding Spike Encoder with Spike Trains (Surrogate Function: ATan) """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoSpikeAtan.apply

        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std

        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        obs = obs.view(-1, self.obs_dim, 1)

        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        
        # Generate Deterministic Spike Trains (Surrogate Function: ATan)
        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt)
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_VTH
        
        return pop_spikes


class PopEncoder(PopSpikeEncoderDeterministic):
    """ Learnable Population Coding Encoder"""
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)

    def forward(self, obs, batch_size):
        obs = obs.view(-1, self.obs_dim, 1)

        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_inputs = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        
        # Generate Input Trains
        for step in range(self.spike_ts):
            pop_inputs[:, :, step] = pop_act
        
        return pop_inputs


class PopDecoder(nn.Module):
    """ Learnable Population Coding Decoder """
    def __init__(self, act_dim, pop_dim, decode, output_activation=nn.Tanh):
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decode = decode
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, pop_act):
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        
        raw_act = self.decoder(pop_act).view(-1, self.act_dim)
        
        if 'fr' in self.decode:
            return self.output_activation(raw_act)
        
        return raw_act


class SpikeMLP(nn.Module):

    def __init__(self, in_pop_dim, act_dim, dec_pop_dim, hidden_sizes, spike_ts, device, decode, neurons, connections):
        super().__init__()

        self.theta_v = -0.172
        self.theta_u = 0.529
        self.theta_r = 0.021
        self.theta_s = 0.132

        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = act_dim * dec_pop_dim
        self.act_dim = act_dim
        self.dec_pop_dim = dec_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.decode = decode
        self.pseudo_spike = PseudoSpikeRect.apply
        self.connections = connections

        # Define Layers (Hidden Layers + Output Population)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0])])
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.extend([nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer])])
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], self.out_pop_dim)
        if connections == 'intra' or connections == 'no-self':
            self.conn = nn.Conv1d(act_dim, self.out_pop_dim, dec_pop_dim, groups=act_dim)
        elif connections == 'no-bias' or connections == 'lateral':
            self.conn = nn.Conv1d(act_dim, self.out_pop_dim, dec_pop_dim, groups=act_dim, bias=False)
        elif connections == 'no-lateral':
            self.weight = nn.Parameter(torch.randn(self.out_pop_dim))
            self.bias = nn.Parameter(torch.randn(self.out_pop_dim))
        elif connections == 'self':
            self.weight = nn.Parameter(torch.randn(self.out_pop_dim))
        elif connections == 'bias':
            self.bias = nn.Parameter(torch.randn(self.out_pop_dim))

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike, resistance, conn_flag=False):
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        if conn_flag:
            if self.connections == 'intra' or self.connections == 'no-bias':
                current = current + self.conn(spike.view(-1, self.act_dim, self.dec_pop_dim)).view(-1, self.out_pop_dim)
            elif self.connections == 'no-self' or self.connections == 'lateral':
                lateral = self.conn.weight.data
                for i in range(self.act_dim):
                    for j in range(self.dec_pop_dim):
                        lateral[i * self.dec_pop_dim + j, 0, j] = 0
                self.conn.weight = nn.Parameter(lateral)
                current = current + self.conn(spike.view(-1, self.act_dim, self.dec_pop_dim)).view(-1, self.out_pop_dim)
            elif self.connections == 'no-lateral':
                current = current + torch.mul(self.weight, spike) + self.bias
            elif self.connections == 'self':
                current = current + torch.mul(self.weight, spike)
            elif self.connections == 'bias':
                current = current + self.bias
                
        volt = volt * (1. - spike) + spike * self.theta_r
        resistance = resistance + spike * self.theta_s
        volt_delta = volt ** 2 - volt - resistance + current
        resistance_delta = self.theta_v * volt - self.theta_u * resistance
        volt = volt + volt_delta
        resistance = resistance + resistance_delta
        spike = self.pseudo_spike(volt)
        return current, volt, spike, resistance

    def forward(self, in_pop_spikes, batch_size):
        hidden_states = []
        for layer in range(self.hidden_num):
            hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                  for _ in range(4)])
        out_pop_spikes = []    
        out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device)
                          for _ in range(4)]

        # Start Spike Timestep Iteration
        for step in range(self.spike_ts):
            in_pop_spike_t = in_pop_spikes[:, :, step]
            hidden_states[0][0], hidden_states[0][1], hidden_states[0][2], hidden_states[0][3] = self.neuron_model(
                self.hidden_layers[0], in_pop_spike_t,
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2], hidden_states[0][3]
            )
            if self.hidden_num > 1:
                for layer in range(1, self.hidden_num):
                    hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2], hidden_states[layer][3] = self.neuron_model(
                        self.hidden_layers[layer], hidden_states[layer-1][2],
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2], hidden_states[layer][3]
                    )
            out_pop_states[0], out_pop_states[1], out_pop_states[2], out_pop_states[3] = self.neuron_model(
                self.out_pop_layer, hidden_states[-1][2],
                out_pop_states[0], out_pop_states[1], out_pop_states[2], out_pop_states[3], True
            )
            out_pop_spikes.append(out_pop_states[2])

        if 'fr' in self.decode:
            out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)
            for spike in out_pop_spikes:
                out_pop_act += spike

            out_pop_act = out_pop_act / self.spike_ts

            return out_pop_act

        return out_pop_spikes


class MDCSpikeActor(nn.Module):
    
    def __init__(self, obs_dim, act_dim, enc_pop_dim, dec_pop_dim, hidden_sizes,
                 mean_range, std, spike_ts, act_limit, device, encode, decode, 
                 v_decay, neurons, lateral):
        super().__init__()
        self.act_limit = act_limit
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.spike_ts = spike_ts
        self.device = device
        self.encode = encode
        self.decode = decode
        self.v_decay = v_decay
        if 'pop' in encode:
            if encode == 'pop-det':
                self.encoder = PopSpikeEncoderDeterministic(obs_dim, enc_pop_dim, spike_ts, mean_range, std, device)
            else:   # 'pop'
                self.encoder = PopEncoder(obs_dim, enc_pop_dim, spike_ts, mean_range, std, device)

            self.snn = SpikeMLP(obs_dim * enc_pop_dim, act_dim, dec_pop_dim, hidden_sizes, spike_ts, device, decode, neurons, connections)

        else: # 'layer'
            self.snn = SpikeMLP(obs_dim, act_dim, dec_pop_dim, hidden_sizes, spike_ts, device, decode, neurons, connections)
        
        self.decoder = PopDecoder(act_dim, dec_pop_dim, decode)

    def forward(self, obs, batch_size):
        if 'pop' in self.encode:
            input_trains = self.encoder(obs, batch_size)
        else:
            input_trains = torch.zeros(batch_size, self.obs_dim, self.spike_ts, device=self.device)
            for step in range(self.spike_ts):
                input_trains[:, :, step] = obs

        out_pop_activity = self.snn(input_trains, batch_size)

        if 'fr' in self.decode:
            return self.act_limit * self.decoder(out_pop_activity)

        v_history = []

        raw_act = torch.zeros(batch_size, self.act_dim, device=self.device)
        for activity in out_pop_activity:
            raw_act = raw_act * self.v_decay + self.decoder(activity)
            v_history.append(raw_act)

        if self.decode == 'max-mem':
            v_stack = torch.stack(v_history, 0)
            max_mem = torch.max(v_stack, 0).values
            min_mem = torch.min(v_stack, 0).values
            mem = max_mem * (max_mem.abs() > min_mem.abs()) + min_mem * (max_mem.abs() <= min_mem.abs())
                
        elif self.decode == 'mean-mem':
            mem = torch.mean(torch.stack(v_history, 0), 0)

        else:   # 'last-mem'
            mem = v_history[-1] 
        
        return self.act_limit * torch.tanh(mem)