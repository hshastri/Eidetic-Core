import torch
import math
import torch.nn as nn
import random
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

class EideticLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out, n_quantile_rate, quantile_cardinality):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        self.outputValues = np.zeros([quantile_cardinality + 1, size_out])
        self.index = 0
        self.n_quantile_rate = n_quantile_rate
        self.quantiles = []
        self.quantile_cardinality = quantile_cardinality

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def calculate_n_quantiles(self, num_quantiles):
        self.outputValues.view('i8,i8,i8,i8,i8,i8,i8,i8,i8,i8').sort(order=['f0'], axis=0)
        
        val = int(self.quantile_cardinality / num_quantiles)

        for i in range(0, num_quantiles):
            print(self.outputValues[val*(i+1)])

        

    def forward(self, x, store_activations, get_indices):
        w_times_x= torch.mm(x, self.weights.t())
        

        all_activations = w_times_x.detach().cpu().numpy()

        for activation_vector in all_activations:
     
            self.outputValues[self.index] = activation_vector
            self.index = self.index + 1

        indices = torch.zeros(self.size_out)

        return [torch.add(w_times_x, self.bias), indices]  

class IndexedLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out, num_indices):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x, store_activations, get_index):
        w_times_x= torch.mm(x, self.weights.t())
        rand = random.uniform(0, 1)


        return torch.add(w_times_x, self.bias) 