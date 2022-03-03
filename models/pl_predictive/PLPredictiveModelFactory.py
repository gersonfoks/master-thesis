'''
THis file contains a class that is a factory that construct models based on some specification
'''
from models.pl_predictive.MSEPredictiveModel import MSEPredictiveModel
from models.pl_predictive.feature_functions import feature_functions, preprocess_functions
from models.predictive.PredictiveModelFactory import PredictiveModelFactory

from utils.parsing.predictive import load_nmt_model
import torch
from torch import nn
import types
import os

activation_functions = {
    'silu': torch.nn.SiLU,
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh
}





class PLPredictiveModelFactory:

    @staticmethod
    def create_model(config, nmt_model=None, tokenizer=None):

        # Load NMT model + tokenizer
        if nmt_model == None or tokenizer == None:
            nmt_model, tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)
        # Load predictive layer
        predictive_model = PLPredictiveModelFactory.create_predictive_model(config)

        # Load the optimizer function

        optimizer_function = lambda x: torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])

        # Construct model
        feature_names = config["features"]
        if config["loss_function"] == "MSE":
            pl_model = MSEPredictiveModel(nmt_model, tokenizer, predictive_model, feature_names, optimizer_function)
        else:
            raise ValueError("Not a known type: {}".format(config["type"]))

        # Set the get_features function
        pl_model.get_features = types.MethodType(feature_functions[config["feature_type"]], pl_model)
        pl_model.preprocess_function = types.MethodType(preprocess_functions[config["preprocess_type"]], pl_model)

        # Return the model
        return pl_model

    @staticmethod
    def create_predictive_model(architecture_config):

        model_factory = PredictiveModelFactory(architecture_config)
        return model_factory.get_model()


