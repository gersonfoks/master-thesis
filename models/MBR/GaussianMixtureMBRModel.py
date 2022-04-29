import numpy as np
import torch

import torch.distributions as td

from models.MBR.BaseMBRModel import BaseMBRModel
from utils.translation_model_utils import batch_sample, batch


class GaussianMixtureMBRModel(BaseMBRModel):
    '''
    Model wraps the NMT model
    '''

    def __init__(self, predictive_model, device="cuda", sample_size=1000):
        super().__init__(predictive_model)
        self.device_name = device
        self.predictive_model = predictive_model.to(self.device_name)
        self.sample_size = sample_size
        self.predictive_model.eval()

    def forward(self, source, n_samples_per_source=256, batch_size=16, ):
        hypotheses = batch_sample(self.predictive_model.nmt_model, self.predictive_model.tokenizer, [source],
                                  n_samples=n_samples_per_source, )

        return self.get_best(source, hypotheses)

    def get_scores(self, sources, samples, batch_size=16):

        model_out = self.get_model_out(sources, samples, batch_size=batch_size)

        scores = self.model_out_to_risk(model_out)

        return scores

    def get_model_out(self, sources, samples, batch_size=16):
        result = {}
        for x, y in zip(batch(sources, n=batch_size), batch(samples, n=batch_size)):
            model_out = self.predictive_model.predict(x, y)
            result = self.add_model_out_to_result(result, model_out)
        result = {k: torch.tensor(np.array(v)) for k, v in result.items()}

        return result

    def get_best(self, source, hypotheses, batch_size=16):
        sources = [source] * len(hypotheses)

        scores = self.get_scores(sources, hypotheses, batch_size=batch_size)

        best_index = np.argmax(scores)

        best = hypotheses[best_index]

        return best

    def model_out_to_risk(self, model_out):
        return self.model_out_to_mean(model_out)

    def get_mean(self, sources, samples, batch_size=16, n_samples=1000):
        model_out = self.get_model_out(sources, samples, batch_size=batch_size)
        mean = self.model_out_to_mean(model_out, n_samples=n_samples)
        return mean

    def model_out_to_mean(self, model_out, n_samples=1000):
        mixture = self.get_mixture(model_out["loc"], model_out["scale"], model_out["logits"])

        sample_scores = mixture.sample((n_samples,))

        mean = sample_scores.mean(0)

        return mean

    def add_model_out_to_result(self, result, model_out):
        if result == {}:
            result["loc"] = []
            result["scale"] = []
            result["logits"] = []
        result['loc'] += list(model_out[0].cpu().numpy())
        result['scale'] += list(model_out[1].cpu().numpy())
        result['logits'] += list(model_out[2].cpu().numpy())

        return result

    def get_mixture(self, loc, scale, logits):
        components = self.make_components(loc, scale,)
        mixture = td.MixtureSameFamily(td.Categorical(logits=logits), components)

        return mixture

    def make_components(self, loc, scale):
        loc = loc.unsqueeze(-1)
        scale = scale.unsqueeze(-1)
        return td.Independent(td.Normal(loc=loc, scale=scale), 1)

    def get_samples(self, sources, hypotheses, n_samples=1000):
        model_out = self.get_model_out(sources, hypotheses)
        mixture = self.get_mixture(model_out["loc"], model_out["scale"], model_out["logits"])

        return mixture.sample((n_samples,))



    # def make_components(self, loc, scale, sample_size):
    #     shape = loc.shape
    #     loc = loc.unsqueeze(-1).repeat((1,) * len(shape) + (sample_size,))
    #     scale = scale.unsqueeze(-1).repeat((1,) * len(shape) + (sample_size,))
    #     return td.Independent(td.Normal(loc=loc, scale=scale), 1)
