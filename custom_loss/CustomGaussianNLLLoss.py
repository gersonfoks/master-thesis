import torch
from torch import nn
import torch.distributions as td


class CustomGaussianNLLLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, loc, utilities, scale):
        loc = loc.unsqueeze(-1)
        scale = scale.unsqueeze(-1) + self.eps


        log_p = td.Independent(td.Normal(loc, scale), 1).log_prob(utilities)
        loss = -log_p.mean(0)
        # Normalize to easier interpret the results
        return loss * 1/utilities.shape[-1]
