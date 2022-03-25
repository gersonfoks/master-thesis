import numpy as np
import torch

from models.MBR_model.BaseMBRModel import BaseMBRModel
from utils.translation_model_utils import batch_sample, batch


class GaussianMBRModel(BaseMBRModel):
    '''
    Model wraps the NMT model
    '''

    def __init__(self, predictive_model, device="cuda", ):
        super().__init__(predictive_model, device="cuda")
        self.device_name = device
        self.predictive_model = predictive_model.to(self.device_name)

    def forward(self, source, n_samples_per_source=256, batch_size=16, ):
        hypotheses = batch_sample(self.predictive_model.nmt_model, self.predictive_model.tokenizer, [source],
                                  n_samples=n_samples_per_source, )

        return self.get_best(source, hypotheses)

    def get_scores(self, sources, samples, batch_size=16):
        scores = []
        for x, y in zip(batch(sources, n=batch_size), batch(samples, n=batch_size)):
            model_out = self.predictive_model.predict(x, y)
            risks = self.model_out_to_risk(model_out)
            scores += list(risks)
        return scores

    def get_model_out(self, sources, samples, batch_size=16):
        result = {}
        for x, y in zip(batch(sources, n=batch_size), batch(samples, n=batch_size)):
            model_out = self.predictive_model.predict(x, y)
            result = self.add_model_out_to_result(result, model_out)
        return result

    def get_best(self, source, hypotheses, batch_size=16):
        sources = [source] * len(hypotheses)

        scores = self.get_scores(sources, hypotheses, batch_size=batch_size)

        best_index = np.argmax(scores)

        best = hypotheses[best_index]

        return best

    # TODO make this more general
    def model_out_to_risk(self, model_out):

        return model_out[0].cpu().numpy().flatten()

    def add_model_out_to_result(self, result, model_out):
        if result == {}:
            result["loc"] = []
            result["scale"] = []
        result['loc'] += list(model_out[0].cpu().numpy().flatten())
        result['scale'] += list(model_out[1].cpu().numpy().flatten())
        return result
