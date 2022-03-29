'''
THis file contains a class that is a factory that construct models based on some specification
'''
import os

import transformers


from models.pl_predictive.GaussianMixturePredictiveModel import GaussianMixturePredictiveModel
from models.pl_predictive.GaussianPredictiveModel import GaussianPredictiveModel
from models.pl_predictive.MSEPredictiveModel import MSEPredictiveModel
from models.pl_predictive.StudentTMixturePredictiveModel import StudentTMixturePredictiveModel
from models.pl_predictive.feature_functions import preprocess_functions, FeatureMap
from models.predictive_head.HeadFactory import HeadFactory

from utils.parsing.predictive import load_nmt_model
import torch

import types

from pathlib import Path

activation_functions = {
    'silu': torch.nn.SiLU,
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh
}


def get_optimizer_function(config):
    print(config["weight_decay"])
    if config["optimizer"] == "adam":
        return lambda x: torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])
    if config["optimizer"] == "adam_with_schedule":
        def f(x):
            optimizer = torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])
            schedular = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                                        num_warmup_steps=7200)

            return [optimizer], [schedular]

        return f


class PLPredictiveModelFactory:

    def __init__(self, config):
        self.config = config

        self.head_factory = None

    def create_model(self, nmt_model=None, tokenizer=None, pretrained_head_path=None):

        # Load NMT model + tokenizer
        if nmt_model == None or tokenizer == None:
            nmt_model, tokenizer = load_nmt_model(self.config["nmt_model"], pretrained=True)
        # Load predictive layer
        feature_map = FeatureMap(self.config["features"])

        feature_names = feature_map.get_feature_names()

        self.config["feature_names"] = feature_names
        head = self.create_head(pretrained_head_path)

        # Load the optimizer function

        optimizer_function = get_optimizer_function(self.config)
        # Construct model
        print(self.config["loss_function"])
        if self.config["loss_function"] == "MSE":
            pl_model = MSEPredictiveModel(nmt_model, tokenizer, head, feature_names, optimizer_function,
                                          feature_map)
        elif self.config["loss_function"] == "gaussian" or self.config["loss_function"] == "gaussian-full":# Second one is legacy
            pl_model = GaussianPredictiveModel(nmt_model, tokenizer, head, feature_names, optimizer_function,
                                               feature_map)

        elif self.config["loss_function"] == "gaussian-mixture":
            print("using a gaussian mixture model")
            pl_model = GaussianMixturePredictiveModel(nmt_model, tokenizer, head, feature_names, optimizer_function,
                                                      feature_map, n_mixtures=self.config["n_mixtures"])
        elif self.config["loss_function"] == "gaussian":
            pl_model = GaussianPredictiveModel(nmt_model, tokenizer, head, feature_names, optimizer_function,
                                               feature_map, )
        elif self.config["loss_function"] == "student-t-mixture":
            print("using a student-t mixture model")
            pl_model = StudentTMixturePredictiveModel(nmt_model, tokenizer, head, feature_names, optimizer_function,
                                               feature_map, )
        else:
            raise ValueError("Not a known type: {}".format(self.config["type"]))

        # Set the get_features function

        pl_model.preprocess_function = types.MethodType(preprocess_functions[self.config["preprocess_type"]], pl_model)

        # Return the model
        return pl_model

    def create_head(self, pretrained_head_path=None):
        '''
        Creates the head and sets the head factory
        :param pretrained_head_path:
        :return:
        '''

        if pretrained_head_path != None:

            head, factory = HeadFactory.load(pretrained_head_path)
            self.head_factory = factory
        else:
            self.head_factory = HeadFactory(self.config)
            head = self.head_factory.get_head()

        return head

    def save(self, pl_model, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        pl_path = path + 'pl_model.pt'

        state = {
            "config": self.config

        }

        torch.save(state, pl_path)

        self.head_factory.save(pl_model.head, path)

    @classmethod
    def load(self, path):
        # Create dir if not exists.

        pl_path = path + 'pl_model.pt'
        checkpoint = torch.load(pl_path)
        factory = PLPredictiveModelFactory(checkpoint["config"])
        model = factory.create_model(pretrained_head_path=path)

        return model, factory
