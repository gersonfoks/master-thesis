from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


from utils.utils import get_all_word_embeddings, nearest_neigbhor

from abc import abstractmethod

'''
This file contains the wrapper for the comet model.
This wrapper is used to make sure we have access to the layers that we need for our experiments
'''


class UtilityWrapper(nn.Module):

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.start_sequence_index = 0
        self.end_sequence_index = 2
        self.padding_index = 1
        self.all_word_embeddings = None
        self.vocab_size = None

    @abstractmethod
    def prepare_sample(self, sample):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, sample):
        raise NotImplementedError()

    @abstractmethod
    def get_word_embeddings(self, input_ids):
        raise NotImplementedError()

    @abstractmethod
    def forward_word_embeddings(self, word_embeddings, sample=None, prepared_sample=None):
        raise NotImplementedError()

    def pad_and_attention(self, tokens, longest_seq):
        # We need to do our own padding and attention to make sure it is consistent with the tokens/embeddings that we use
        if longest_seq - len(tokens) > 0:
            padded_tokens = tokens + [self.padding_index] * (longest_seq - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * (longest_seq - len(tokens))
        else:
            padded_tokens = tokens
            attention_mask = [1] * len(tokens)
        return padded_tokens, attention_mask

    def add_start_end_token(self, word_embedding):
        start_token = torch.tensor([[self.start_sequence_index]]).to(self.device)
        end_token = torch.tensor([[self.end_sequence_index]]).to(self.device)

        start_embedding = self.get_word_embeddings(start_token)
        end_embedding = self.get_word_embeddings(end_token)
        return torch.cat([start_embedding, word_embedding, end_embedding], dim=1)

    def prepare_embedding_sample(self, word_embeddings, sample):
        # Get the tokens for the word_embeddings
        tokens = self.word_embeddings_to_tokens(word_embeddings)

        prepared_sample = self.prepare_tokens_sample(tokens, sample)
        # Lastly we need to pad the embeddings

        attention_mask = prepared_sample["hypothesis"]["attention_mask"][0]

        padding_length = len(attention_mask) - sum(attention_mask)

        if padding_length > 0:
            input_ids = torch.tensor([[self.padding_index]*  padding_length] ).to(self.device)
            padding = self.get_word_embeddings(input_ids)

            word_embeddings = torch.cat([word_embeddings, padding], dim=1)
        prepared_sample['hypothesis']["word_embeddings"] = word_embeddings
        return prepared_sample
    def word_embeddings_to_tokens(self, word_embeddings):
        return self.nearest_neighbors(word_embeddings.squeeze(dim=0))

    def nearest_neighbors(self, vectors):
        if self.all_word_embeddings == None:
            self.initialize_all_word_embeddings()

        indices = []
        for vector in vectors:
            indices.append(nearest_neigbhor(vector, self.all_word_embeddings).item())
        return indices

    def initialize_all_word_embeddings(self):
        self.all_word_embeddings = get_all_word_embeddings(self.word_embeddings, device=self.device)

    def logits_to_sentence(self, logits, ):

        tokens = torch.argmax(logits, dim=-1).squeeze(dim=0).detach().cpu().numpy()

        result = []
        for token in tokens:
            word = self.tokenizer.decode(token)

            result.append(word)
        sentence = " ".join(result)

        return sentence

    def get_random_embedding(self, num_words):
        random_tokens = self.get_random_tokens(num_words)
        return self.word_embeddings(random_tokens)

    def get_random_tokens(self, num_tokens):
        # We do not use the first 3 tokens as they are special tokens
        return torch.randint(low=3, high=self.vocab_size, size=(1, num_tokens,)).long()



    def forward_logits(self, logits, sample):
        logit_sample = {"logits": logits, "refs": sample["refs"], "src": sample["src"]}
        prepared_logit_sample = self.prepare_logits_sample(logit_sample)
        word_embeddings = prepared_logit_sample["hypothesis"]["word_embeddings"]
        return self.forward_word_embeddings(word_embeddings, prepared_sample=prepared_logit_sample)

    def embedding_to_sentence(self, word_embeddings):
        tokens = self.word_embeddings_to_tokens(word_embeddings)

        return self.tokenizer.decode(tokens)
