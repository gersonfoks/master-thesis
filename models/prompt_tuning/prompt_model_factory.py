
from models.misc import activation_functions
from models.pl_predictive.PLPredictiveModelFactory import get_optimizer_function
from models.prompt_tuning.PlPromptModel import PLPromptModel

from utils.parsing.predictive import load_nmt_model
import torch
from torch import nn
import numpy as np



class PromptModelFactory:

    def __init__(self, config):
        self.config = config

        self.head_factory = None

    def create_model(self, nmt_model=None, tokenizer=None, pretrained_head_path=None):

        # Load NMT model + tokenizer
        if nmt_model == None or tokenizer == None:
            nmt_model, tokenizer = load_nmt_model(self.config["nmt_model"], pretrained=True)
        # Load predictive layer

        head = self.create_head(pretrained_head_path)

        # Load the optimizer function

        optimizer_function = get_optimizer_function(self.config)


        encoder_prompt_embedding, decoder_prompt_embedding = self.get_prompt_embedding(nmt_model, tokenizer, self.config["n_prompts"])

        prompt_model = PLPromptModel(nmt_model, tokenizer, head, encoder_prompt_embedding, decoder_prompt_embedding, optimizer_function)

        return prompt_model

    def get_prompt_embedding(self, nmt_model, tokenizer, n_prompts):

        encoder_ids = torch.tensor(np.random.choice(tokenizer.vocab_size, n_prompts))
        decoder_ids = torch.tensor(np.random.choice(tokenizer.vocab_size, n_prompts))

        encoder_embeds = nmt_model.model.encoder.embed_tokens(encoder_ids)
        decoder_embeds = nmt_model.model.decoder.embed_tokens(decoder_ids)
        encoder_prompt = torch.nn.Parameter(encoder_embeds)
        decoder_prompt = torch.nn.Parameter(decoder_embeds)
        encoder_prompt.requires_grad = True
        decoder_prompt.requires_grad = True
        return encoder_prompt, decoder_prompt


    def create_head(self, pretrained_head_path=None):
        '''
        Creates the head and sets the head factory
        :param pretrained_head_path:
        :return:
        '''

        if pretrained_head_path != None:
            head = None
        else:
            head = self.get_predictive_layers()

        return head

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

    # def save(self, pl_model, path):
    #     Path(path).mkdir(parents=True, exist_ok=True)
    #     pl_path = path + 'pl_model.pt'
    #
    #     head_path = path + 'head.pt'
    #
    #     state = {
    #         "config": self.config,
    #
    #
    #     }
    #
    #     torch.save(state, pl_path)
    #
    #     self.head_factory.save(pl_model.head, path)
    #
    # def load_head(self, head_path):
    #     pass
    #
    # @classmethod
    # def load(self, path):
    #     # Create dir if not exists.
    #
    #     pl_path = path + 'pl_model.pt'
    #     checkpoint = torch.load(pl_path)
    #     factory = PromptModelFactory(checkpoint["config"])
    #     model = factory.create_model(pretrained_head_path=path)
    #
    #     return model, factory
