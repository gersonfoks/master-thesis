from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.pools_utils import max_pooling, average_pooling

# A wrapper for comet (makes predicting faster then when using as described in the documentation of comet.
# ( This is due predicting with model creates a PytorchLightning trainer which costs time and gives an error for some reason whe )
class CometWrapper:

    def __init__(self, cometModel, device='cuda'):
        super().__init__()

        self.model = cometModel
        self.model.caching = False
        print("uses caching")
        print(self.model.caching)
        self.device = device

    def predict(self, samples):
        prepared_samples = self.model.prepare_for_inference(samples, )
        inputs = {k: v.to(self.device) for k,v in prepared_samples.items()}
        return self.model.forward(**inputs)

