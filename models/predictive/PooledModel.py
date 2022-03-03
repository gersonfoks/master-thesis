'''
    This model does cross attention on layers
'''

import torch

from models.predictive.PredictiveBaseModel import BasePredictiveModel


class PooledModel(BasePredictiveModel):

    def __init__(self, predictive_layers, feature_names):
        '''

        :param layers: the size of the layers
        :param feature_names:
        :param attention_names:
        '''
        super().__init__()

        self.predictive_layers = predictive_layers
        self.feature_names = feature_names

    def forward(self, features):
        '''
        Forward a list of features. Expect to have the following form
        :param features:
        :return:
        '''

        feature_list = [
            features[name] for name in self.feature_names
        ]
        cat_features = torch.cat(feature_list, dim=-1)

        return self.predictive_layers(cat_features)
