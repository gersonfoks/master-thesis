'''
    This model does cross attention on layers
'''

from torch import nn
import torch

from models.estimation.pool_utils import avg_pooling, max_pooling


class PooledHead(nn.Module):

    def __init__(self,  predictive_layers, feature_names):
        '''

        :param layers:
        :param feature_names:
        :param attention_names:
        '''
        super().__init__()

        self.predictive_layers = predictive_layers

        # Used for features
        self.feature_names = feature_names


    def forward(self, features):
        '''
        Forward a list of features. Expect to have the following form
        :param features:
        :return:
        '''
        temp = []


        for feature_name in self.feature_names:
            feature_in = features[feature_name] # Forward layers


            padding_mask = ~features[feature_name + '_mask'].bool()


            average_features = avg_pooling(feature_in, padding_mask)
            max_features = max_pooling(feature_in, padding_mask)

            temp.append(average_features)
            temp.append(max_features)


        predictive_layer_input = torch.concat(temp, dim=-1)



        return self.predictive_layers(predictive_layer_input)

    def get_risk(self, features):
        pass
