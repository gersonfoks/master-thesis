import torch
from torch.nn import Module


class GaussianMixtureLoss(Module):

    def __init__(self, n_mixtures=2):
        super().__init__()

        self.n_mixtures = n_mixtures
        self.gaussian_loss = torch.nn.GaussianNLLLoss(full=True, reduction='none')

    def forward(self, averages, targets, vars, weights):
        loss = 0
        for i, (average, var) in enumerate(zip(averages, vars)):
            gauss_loss = self.gaussian_loss(average, targets, var, )

            loss += weights[:, i] * gauss_loss

        loss = torch.mean(loss)
        return loss
