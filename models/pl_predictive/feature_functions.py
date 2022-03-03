from models.pl_predictive.pool_utils import avg_pooling, max_pooling
import torch

import numpy as np


def get_pooled_features(self, input_ids, attention_mask, labels, decoder_input_ids):
    nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                     decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                     output_attentions=True)
    encoder_last_hidden_state = nmt_out["encoder_last_hidden_state"]
    decoder_last_hidden_state = nmt_out["decoder_hidden_states"][-1]

    # Next perform average pooling
    # first apply attention_mask to encoder_last_hidden_state
    attention_mask_decoder = (self.padding_id != labels).long()

    avg_encoder_hidden_state = avg_pooling(encoder_last_hidden_state, attention_mask)

    avg_decoder_hidden_state = avg_pooling(decoder_last_hidden_state, attention_mask_decoder)

    max_encoder_hidden_state = max_pooling(encoder_last_hidden_state, attention_mask)

    max_decoder_hidden_state = max_pooling(decoder_last_hidden_state, attention_mask_decoder)

    return {"avg_pool_encoder_hidden_state": avg_encoder_hidden_state,
            "avg_pool_decoder_hidden_state": avg_decoder_hidden_state,
            "max_pool_encoder_hidden_state": max_encoder_hidden_state,
            "max_pool_decoder_hidden_state": max_decoder_hidden_state,
            }


def get_last_layer_features(self, input_ids, attention_mask, labels, decoder_input_ids):
    nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                     decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                     output_attentions=True)
    encoder_last_hidden_state = nmt_out["encoder_last_hidden_state"]
    decoder_last_hidden_state = nmt_out["decoder_hidden_states"][-1]

    attention_mask_decoder = (self.padding_id != labels).long()

    return {"encoder_last_hidden_state": (encoder_last_hidden_state, attention_mask),
            "decoder_last_hidden_state": (decoder_last_hidden_state, attention_mask_decoder),
            }


def preprocess_function_pooled(self, batch):
    '''
    Function that preprocesses the dataset
    :param batch:
    :return:
    '''
    sources = batch["sources"]
    hypotheses = batch["hypothesis"]
    with torch.no_grad():
        features = self.get_features_batch(sources, hypotheses)

    features = {k: v.cpu().numpy() for k, v in features.items()}

    return features


def apply_attention(layer, attention):
    '''
    Removes the vectors that are not attended to (entries where attention = 0) and returns the numpy array
    :param layer:
    :param attention:
    :return:
    '''
    layer = layer.cpu().numpy()
    attention = attention.cpu().numpy()

    lengths = np.argmin(attention, axis=-1)

    # Slice everything, if lengths = 0 we need to take the whole array
    result = [l[:i] if i > 0 else l for l, i in zip(layer, lengths)]

    return result


def preprocess_function_full(self, batch):
    '''
        Function that preprocesses the dataset
        :param batch:
        :return:
        '''
    sources = batch["sources"]
    hypotheses = batch["hypothesis"]
    with torch.no_grad():
        features = self.get_features_batch(sources, hypotheses)

    features = {k: apply_attention(hidden_layer, attention_mask) for k, (hidden_layer, attention_mask) in
                features.items()}

    return features


preprocess_functions = {
    'pooled': preprocess_function_pooled,
    'full': preprocess_function_full,
}

feature_functions = {
    'last_layer_pool': get_pooled_features,
    'last_layer': get_last_layer_features,
}
