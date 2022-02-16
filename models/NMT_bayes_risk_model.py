import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np

from utils.translation_model_utils import batch_sample, batch


class NMTBayesRisk(nn.Module):

    def __init__(self, model):
        super(NMTBayesRisk, self).__init__()

        self.model = model
        self.nmt_model = model.nmt_model

        self.tokenizer = model.tokenizer
        self.nmt_model.eval()

    def forward(self, source, n_samples_per_source=32, batch_size=16, score_type="avg-std"):
        '''
        IMPORTANT: NAIVE implementation, only allows for one source at the same time!
        :param input_ids:
        :param attention_mask:
        :param n_samples_per_source:
        :return:
        '''
        # First we sample

        samples = batch_sample(self.nmt_model, self.tokenizer, [source], n_samples=n_samples_per_source)

        # Then we predict
        sources = [source] * n_samples_per_source
        scores = []
        for x,y in zip(batch(sources, n=batch_size), batch(samples, n=batch_size)):
            new_scores = self.model.predict(x,y).cpu().numpy().flatten()
            scores += list(new_scores)
            # if score_type == "avg-std":
            #     scores += list(avg.cpu().numpy().flatten() - std.cpu().numpy().flatten())
            # elif score_type == "avg":
            #     scores += list(avg.cpu().numpy().flatten())
            # else:
            #     raise ValueError("not known score type: {}".format(score_type))


        best_index = np.argmax(scores)
        best = samples[best_index]


        # Lastly we return the best one:
        return best
