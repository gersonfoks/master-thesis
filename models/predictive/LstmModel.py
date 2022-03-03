'''
    This model does cross attention on layers
'''

import torch

from models.predictive.PredictiveBaseModel import BasePredictiveModel


class LSTMModel(BasePredictiveModel):

    def __init__(self, lstm_layers, predictive_layers, feature_names):
        '''

        :param layers: the size of the layers
        :param feature_names:
        :param attention_names:
        '''
        super().__init__()

        self.lstm_layers = lstm_layers
        self.predictive_layers = predictive_layers
        self.feature_names = feature_names

    def forward(self, features):
        '''
        Forward a list of features.
        There should be
        :param features:
        :return:
        '''
        # feature_list = [
        #     features[name] for name in self.feature_names
        # ]
        #

        hidden_states = [
            self.get_hidden_state_out(name, features) for name in self.feature_names
        ]

        # Concate the outputs
        hidden_states_cat = torch.cat(hidden_states, dim=-1)

        return self.predictive_layers(hidden_states_cat)

    def get_hidden_state_out(self, name, x):
        output, hn = self.lstm_layers[name].forward(x[name])
        return hn
