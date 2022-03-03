import torch
from torch import nn


class BasePredictiveModel(nn.Module):

    def get_risk(self, features):
        '''
        Predicts the risk given the features
        :param features:
        :return:
        '''
        raise NotImplementedError()
