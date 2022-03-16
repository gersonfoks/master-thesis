import torch

from torch import nn

from models.predictive_head.AttentionHead import AttentionHead
from models.predictive_head.CrossAttentionHead import CrossAttentionHead
from models.predictive_head.QueryHead import QueryHead
from models.predictive_head.PooledHead import PooledHead
from models.predictive_head.PooledHead import PooledHead

activation_functions = {
    'silu': torch.nn.SiLU,
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh
}


class HeadFactory:

    def __init__(self, config):
        self.config = config

        self.model_functions = {
            "feed_forward": self.create_pooled_model,
            "query": self.create_query_model,
            "self_attention": self.create_attention_model,
            "cross_attention": self.create_cross_attention_model,
            "pooled": self.create_pooled_model,

        }

    def get_head(self):
        return self.model_functions[self.config["model_type"]]()

    def create_pooled_model(self):

        config = self.config

        predictive_layers = self.get_predictive_layers()

        return PooledHead(predictive_layers, config["feature_names"])

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

        return nn.Sequential(*layers)

    def create_query_model(self):

        predictive_layers = self.get_predictive_layers()

        query_layers, queries = self.create_query_layers()

        return QueryHead(query_layers, predictive_layers, queries, self.config["feature_names"])

    def create_query_layers(self, alternative_names=None):

        # Use default names unless we overwrite.
        feature_names = self.config["feature_names"] if alternative_names == None else alternative_names

        query_layers = {
            feature_name: nn.MultiheadAttention(512, self.config["query_layers"]["n_heads"],
                                                dropout=self.config["dropout"],
                                                batch_first=True) for
            feature_name in feature_names}

        queries = {}
        for feature_name in feature_names:
            n_queries = self.config["query_layers"]["n_queries"]
            query = nn.Parameter(torch.rand(size=(1, n_queries, 512)))

            torch.nn.init.normal_(query)

            queries[feature_name] = query

        return query_layers, queries

    def create_attention_model(self):

        attention_layers = self.create_attention_layers()

        query_layers, queries = self.create_query_layers()

        predictive_layers = self.get_predictive_layers()

        return AttentionHead(attention_layers, query_layers, queries, predictive_layers, self.config["feature_names"])

    def create_attention_layers(self):
        attention_layers = {
            feature_name: nn.MultiheadAttention(512, self.config["attention_layers"]["n_heads"],
                                                dropout=self.config["dropout"],
                                                batch_first=True) for
            feature_name in self.config["feature_names"]}
        return attention_layers

    def create_cross_attention_layers(self):
        names = self.get_cross_attention_names()
        attention_layers = {
            name: nn.MultiheadAttention(512, self.config["cross_attention_layers"]["n_heads"],
                                        dropout=self.config["dropout"],
                                        batch_first=True) for name in names
        }
        return attention_layers

    def get_cross_attention_names(self):
        return [
            "{}_{}".format(feature_names[0], feature_names[1]) for feature_names in
            self.config["cross_attention_layers"]["cross_attention"]
        ]

    def create_cross_attention_model(self):


        attention_layers = self.create_cross_attention_layers()

        layer_names = self.get_cross_attention_names()

        cross_features = [(cross_feature[0], cross_feature[1]) for cross_feature in
                          self.config["cross_attention_layers"]["cross_attention"]]

        query_layers, queries = self.create_query_layers(alternative_names=layer_names)

        predictive_layers = self.get_predictive_layers()

        return CrossAttentionHead(attention_layers, query_layers, queries, predictive_layers,
                                  self.config["feature_names"], cross_features)

    def save(self, head, path):
        head_path = path + 'head.pt'
        state = {
            'config': self.config,
            'state_dict': head.state_dict()

        }

        torch.save(state, head_path)

    @classmethod
    def load(self, path):
        '''
        Loads the head, returns a tuple: the head and an instantiation of the factory that created the model
        :param path:
        :return:
        '''
        head_path = path + 'head.pt'
        checkpoint = torch.load(head_path)
        factory = HeadFactory(checkpoint["config"])
        head = factory.get_head()
        head.load_state_dict(checkpoint["state_dict"])
        return head, factory
