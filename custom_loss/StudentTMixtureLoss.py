# Inspiration from:

from torch import nn
import torch.distributions as td


class StudentTMixtureLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, df, loc, utilities, scale, logits):
        components = self.make_components(df, loc, scale, utilities.shape[-1])

        mixture = td.MixtureSameFamily(td.Categorical(logits=logits), components)

        log_p = mixture.log_prob(utilities)
        loss = -log_p.mean(0)
        # Divide by the number of independent samples to make the interpretation of the results easier
        return loss / utilities.shape[-1]

    def make_components(self, df, loc, scale, sample_size):
        shape = loc.shape
        df = df.unsqueeze(-1).repeat((1,) * len(shape) + (sample_size,))
        loc = loc.unsqueeze(-1).repeat((1,) * len(shape) + (sample_size,))
        scale = scale.unsqueeze(-1).repeat((1,) * len(shape) + (sample_size,))
        return td.Independent(td.StudentT(df, loc=loc, scale=scale), 1)
