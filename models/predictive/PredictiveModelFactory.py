import torch


from torch import nn

from models.predictive.PooledModel import PooledModel

activation_functions = {
    'silu': torch.nn.SiLU,
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh
}



class PredictiveModelFactory:




    def __init__(self, config):
        self.config = config

        self.model_functions ={
            "feed_forward": self.create_pooled_model,
            "lstm": self.create_lstm_model,
            "attention": self.create_attention_model,
        }

    def get_model(self):
        return self.model_functions[self.config["model_type"]]()

    def create_pooled_model(self):

        config = self.config

        layers = self.get_predictive_layers()

        return PooledModel(nn.Sequential(*layers), config["features"])

    def create_lstm_model(self):
        pass


    def get_predictive_layers(self):
        config = self.config
        activation_function = activation_functions[config['activation_function']]

        layers = []
        # Add all the layers except the last one
        for layer_in, layer_out in zip(config['predictive_layers'][:-2], config['predictive_layers'][1:-1]):
            layers.append(nn.Linear(layer_in, layer_out))
            layers.append(activation_function())
            if config["dropout"] > 0:
                layers.append(nn.Dropout(p=config["dropout"]))

        # Add the last one
        layers.append(nn.Linear(config['predictive_layers'][-2], config['predictive_layers'][-1]))

        return layers

    def create_attention_model(self):
        pass