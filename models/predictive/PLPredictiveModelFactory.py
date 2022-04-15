'''
THis file contains a class that is a factory that construct models based on some specification
'''

from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from models.predictive.GaussianMixturePredictiveModel import GaussianMixturePredictiveModel
from models.predictive.GaussianMixtureSharedPredictiveModel import GaussianMixtureSharedPredictiveModel
from models.predictive.GaussianPredictiveModel import GaussianPredictiveModel
from models.predictive.MSEPredictiveModel import MSEPredictiveModel
from models.predictive.ReferenceMSEPredictiveModel import ReferenceMSEPredictiveModel
from models.predictive.StudentTMixturePredictiveModel import StudentTMixturePredictiveModel
from models.predictive.feature_functions import preprocess_functions, FeatureMap
from models.predictive_head.HeadFactory import HeadFactory

from misc.parsing.predictive import load_nmt_model
import torch

import types

from pathlib import Path

activation_functions = {
    'silu': torch.nn.SiLU,
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh
}


def get_optimizer_function(config):

    if config["optimizer"] == "adam":
        def initializer(x):
            lr_config = {
                "optimizer": torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"]),

            }
            return lr_config

        return initializer
    if config["optimizer"] == "adam_with_schedule":

        def initializer(x):

            num_warmup_steps = config["warmup_steps"]

            optimizer = torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])

            # When to start the decay
            start_step_decay = config["start_decay"]

            def lr_lambda(current_step: int):

                if current_step <= num_warmup_steps:
                    return current_step / num_warmup_steps
                # Waiting a number of steps before decaying
                elif current_step <= start_step_decay:
                    return 1.0
                else:
                    return (current_step - start_step_decay) ** (-0.5)

            lr_config = {
                "optimizer": optimizer,
                "lr_scheduler": {

                    "scheduler": LambdaLR(optimizer, lr_lambda),
                    "interval": "step",
                }

            }

            return lr_config

        return initializer

    if config["optimizer"] == "adam_with_plateau":
        def initializer(x):
            optimizer = torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])

            lr_config = {
                "optimizer": optimizer,
                "lr_scheduler": {

                    "scheduler": ReduceLROnPlateau(optimizer, "min", patience=config["patience"]),

                    "monitor": 'val_loss'
                }

            }

            return lr_config

        return initializer


class PLPredictiveModelFactory:

    def __init__(self, config):
        self.config = config

        self.head_factory = None

    def create_model(self, nmt_model=None, tokenizer=None, pretrained_head_path=None):

        # Load NMT model + tokenizer
        if nmt_model == None or tokenizer == None:
            nmt_model, tokenizer = load_nmt_model(self.config["nmt_model"], pretrained=True)
        # Load the feature map (as different models can use different features)
        feature_map = FeatureMap(self.config["features"])
        feature_names = feature_map.get_feature_names()

        self.config["feature_names"] = feature_names
        head = self.create_head(pretrained_head_path)

        # Load the optimizer function
        optimizer_function = get_optimizer_function(self.config)
        # Construct model

        if self.config["model_type"] == "reference_model":
            print("using reference model")
            pl_model = ReferenceMSEPredictiveModel(nmt_model, tokenizer, head, feature_names, optimizer_function)
        else:

            if self.config["loss_function"] == "MSE":
                pl_model = MSEPredictiveModel(nmt_model, tokenizer, head, feature_names, optimizer_function,
                                              feature_map)
            elif self.config["loss_function"] == "gaussian" or self.config[
                "loss_function"] == "gaussian-full":  # Second one is legacy
                pl_model = GaussianPredictiveModel(nmt_model, tokenizer, head, feature_names, optimizer_function,
                                                   feature_map)

            elif self.config["loss_function"] == "gaussian-mixture":
                print("using a gaussian mixture model")
                if "shared_params" in self.config.keys() and self.config["shared_params"]:
                    print("use shared params")
                    pl_model = GaussianMixtureSharedPredictiveModel(nmt_model, tokenizer, head, feature_names,
                                                                    optimizer_function,
                                                                    feature_map, n_mixtures=self.config["n_mixtures"])
                else:
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
        if self.config["model_type"] != "reference_model":
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
