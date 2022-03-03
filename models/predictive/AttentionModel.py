'''
    This model does cross attention on layers
'''

from torch import nn
import torch



class AttentionModel(nn.Module):


    def __init__(self, attention_layers, predictive_layers, feature_names):
        '''

        :param layers: the size of the layers
        :param feature_names:
        :param attention_names:
        '''
        super().__init__()
        self.attention_layers = attention_layers
        self.predictive_layers = predictive_layers


        # Used for features
        self.feature_names = feature_names
        self.attention_names = ['{}_attention'.format(name) for name in feature_names]



    def forward(self, features):
        '''
        Forward a list of features. Expect to have the following form
        :param features:
        :return:
        '''
        pass


    def get_risk(self, features):
        pass




