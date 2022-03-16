import numpy as np
import torch

from utils.translation_model_utils import batch_sample, batch


class MBRModel(torch.nn.Module):
    '''
    Model wraps the NMT model
    '''

    def __init__(self, predictive_model, device="cuda", ):
        super().__init__()
        self.device_name = device
        self.predictive_model = predictive_model.to(self.device_name)

    def forward(self, source, n_samples_per_source=256, batch_size=16, ):
        samples = batch_sample(self.predictive_model.nmt_model, self.predictive_model.tokenizer, [source],
                               n_samples=n_samples_per_source, )


        sources = [source] * n_samples_per_source
        scores = []
        for x, y in zip(batch(sources, n=batch_size), batch(samples, n=batch_size)):
            risks = self.predictive_model.predict(x, y).cpu().numpy().flatten()
            scores += list(risks)


        best_index = np.argmax(scores)
        print(" samples: ", samples)
        print("risks: ", scores)
        best = samples[best_index]
        print("best:", best)
        # Lastly we return the best one:
        return best
