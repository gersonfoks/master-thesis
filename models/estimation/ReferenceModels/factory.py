import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, StepLR

from misc.parsing.predictive import load_nmt_model

from models.estimation.ReferenceModels.Layers import TokenizerLayer, EmbeddingLayer, PoolLayer, \
    AttentionFeatureExtractionLayer, LSTMFeatureExtractionLayer, LastHiddenStateEmbedding, PoolFeatureExtractionLayer
from models.estimation.ReferenceModels.ReferenceModel import ReferenceModel

activation_functions = {
    'silu': torch.nn.SiLU,
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh,
    'sigmoid': torch.nn.Sigmoid
}



def get_optimizer_function(config):

    if config["optimizer"]["type"] == "adam":
        def initializer(x):
            lr_config = {
                "optimizer": torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"]),

            }
            return lr_config

        return initializer
    if config["optimizer"]["type"] == "adam_with_warmup":

        def initializer(x):

            num_warmup_steps = config["optimizer"]["warmup_steps"]

            optimizer = torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])

            # When to start the decay
            start_step_decay = config["optimizer"]["start_decay"]

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

    if config["optimizer"]["type"] == "adam_with_steps":

        def initializer(x):


            optimizer = torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])
            step_size = config["optimizer"]["step_size"]
            gamma = config["optimizer"]["gamma"]
            # When to start the decay

            lr_config = {
                "optimizer": optimizer,
                "lr_scheduler": {

                    "scheduler": StepLR(optimizer, step_size=step_size, gamma=gamma),
                    "interval": "epoch",
                }

            }

            return lr_config

        return initializer





class ReferenceModelFactory:


    def __init__(self, config):
        self.config = config


    def create_model(self):

        # We first load the nmt model

        self.nmt_model, self.tokenizer = load_nmt_model(self.config["nmt_model"], pretrained=True)

        # Create the different layers
        tokenizer_layer = self.create_tokenizer_layer()
        embedding_layer = self.create_embedding_layer()
        feature_extraction_layer = self.create_feature_extraction_layer()
        final_layers = self.create_final_layer()

        optimizer_function = get_optimizer_function(self.config)
        return ReferenceModel(tokenizer_layer, embedding_layer, feature_extraction_layer, final_layers, optimizer_function)

    def create_tokenizer_layer(self):
        return TokenizerLayer(self.tokenizer)

    def create_embedding_layer(self):

        if self.config["embedding"] == "nmt_embedding":

            return EmbeddingLayer(self.nmt_model.model.encoder.embed_tokens,self.nmt_model.model.decoder.embed_tokens)
        elif self.config["embedding"] == "last_hidden":

            return LastHiddenStateEmbedding(self.nmt_model)




    def create_feature_extraction_layer(self):

        if self.config["feature_extraction_layer"]["type"] == "attention":

            n_heads = self.config["feature_extraction_layer"]["n_heads"]
            dropout = self.config["dropout"]
            source_attention = nn.MultiheadAttention(512, n_heads,
                                  dropout=self.config["dropout"],
                                  batch_first=True)
            target_attention = nn.MultiheadAttention(512, n_heads,
                                                     dropout=dropout,
                                                     batch_first=True)

            source_pool = PoolLayer(512, n_heads, dropout=dropout, batch_first=True)

            target_pool = PoolLayer(512, n_heads, dropout=dropout, batch_first=True)

            return AttentionFeatureExtractionLayer(source_attention, target_attention, source_pool, target_pool)
        elif self.config["feature_extraction_layer"]["type"] == "lstm":

            lstm_source = nn.LSTM(512, 512, batch_first=True)
            lstm_target = nn.LSTM(512, 512, batch_first=True)

            return LSTMFeatureExtractionLayer(lstm_source, lstm_target)
        elif self.config["feature_extraction_layer"]["type"] == "pool":
            n_heads = self.config["feature_extraction_layer"]["n_heads"]
            dropout = self.config["dropout"]


            source_pool = PoolLayer(512, n_heads, dropout=dropout, batch_first=True)

            target_pool = PoolLayer(512, n_heads, dropout=dropout, batch_first=True)

            return PoolFeatureExtractionLayer(source_pool, target_pool)







    def create_final_layer(self):
        config = self.config
        activation_function = activation_functions[config['activation_function']]

        layers = []
        # Add all the layers except the last one
        for layer_in, layer_out in zip(config['feed_forward_layers'][:-2], config['feed_forward_layers'][1:-1]):
            layers.append(nn.Linear(layer_in, layer_out))
            layers.append(activation_function())
            if config["dropout"] > 0:
                layers.append(nn.Dropout(p=config["dropout"]))

        # Add the last one
        layers.append(nn.Linear(config['feed_forward_layers'][-2], config['feed_forward_layers'][-1]))
        activation_function_last_layer = activation_functions[config['activation_function_last_layer']]
        layers.append(activation_function_last_layer())
        return nn.Sequential(*layers)


    def save(self):
        pass


    @classmethod
    def load(self, path):
        pass