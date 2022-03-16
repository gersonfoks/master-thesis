'''
    This model does cross attention on layers
'''

from torch import nn
import torch


class QueryHead(nn.Module):

    def __init__(self, query_layers, predictive_layers, queries, feature_names):
        '''

        :param layers:
        :param feature_names:
        :param attention_names:
        '''
        super().__init__()
        self.query_layers = query_layers
        self.predictive_layers = predictive_layers

        self.queries = queries

        # Used for features
        self.feature_names = feature_names
        self.attention_names = ['{}_attention'.format(name) for name in feature_names]

        #
        # As we use dicts we still need to register.
        for name, param in self.queries.items():
            self.register_parameter(name, param=param)

        self.attention_layers_list = nn.ModuleList([*self.query_layers.values()])

    def forward(self, features):
        '''
        Forward a list of features. Expect to have the following form
        :param features:
        :return:
        '''
        temp = []

        for feature_name in self.feature_names:
            attention_in = features[feature_name]
            padding_mask = features[feature_name + '_mask'].bool()

            queries = self.queries[feature_name].repeat(attention_in.shape[0], 1, 1)

            attention_out, _ = self.query_layers[feature_name](queries, attention_in, attention_in,
                                                               key_padding_mask=padding_mask)

            temp.append(attention_out.reshape(attention_out.shape[0], -1))  # Squeeze everything together.

        predictive_layer_input = torch.concat(temp, dim=-1)

        return self.predictive_layers(predictive_layer_input)

