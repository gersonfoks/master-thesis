import torch
activation_functions = {
    'silu': torch.nn.SiLU,
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh,
    'sigmoid': torch.nn.Sigmoid,
    'none': lambda x: x, # If no activation function.
}
