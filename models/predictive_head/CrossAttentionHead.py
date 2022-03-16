'''
    This model does cross attention on layers
'''

from torch import nn
import torch


class CrossAttentionHead(nn.Module):

    def __init__(self, cross_attention_layers, query_layers, queries, predictive_layers, feature_names, cross_features):
        '''

        :param layers:
        :param feature_names:
        :param attention_names:
        '''
        super().__init__()
        self.cross_attention_layers = cross_attention_layers
        self.query_layers = query_layers
        self.predictive_layers = predictive_layers

        self.queries = queries

        # Used for features
        self.feature_names = feature_names
        self.cross_attention_names = ['{}_attention'.format(name) for name in feature_names]

        self.cross_features = cross_features

        #
        # As we use dicts we still need to register.
        for name, param in self.queries.items():
            self.register_parameter(name, param=param)

        self.module_list = nn.ModuleList([*self.cross_attention_layers.values(), *self.query_layers.values()])

    def forward(self, features):
        '''
        Forward a list of features. Expect to have the following form
        :param features:
        :return:
        '''
        query_layers_out = []

        for (feature_query, feature_key_value) in self.cross_features:
            query_in = features[feature_query]
            attention_in = features[feature_key_value]
            padding_mask_cross = features[feature_key_value + '_mask'].bool()
            padding_mask = features[feature_query + '_mask'].bool()

            name = '{}_{}'.format(feature_query, feature_key_value)

            # First do self attention
            attention_out, _ = self.cross_attention_layers[name](query=query_in, key=attention_in, value=attention_in,
                                                                 key_padding_mask=padding_mask_cross)

            # Then do the query_layers

            queries = self.queries[name].repeat(attention_in.shape[0], 1, 1)
            query_out, _ = self.query_layers[name](queries, attention_out, attention_out,
                                                   key_padding_mask=padding_mask)

            query_layers_out.append(query_out.reshape(attention_out.shape[0], -1))

            # attention_layer_out.append(attention_out.reshape(attention_out.shape[0], -1))  # Squeeze everything together.

        predictive_layer_input = torch.concat(query_layers_out, dim=-1)

        return self.predictive_layers(predictive_layer_input)

    def get_risk(self, features):
        pass
