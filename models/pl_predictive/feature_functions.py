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


class FeatureMap:

    def __init__(self, features, padding_id=-100):
        self.padding_id = padding_id
        self.features = features
        self.functions = {
            "encoder_hidden_state": self.get_encoder_last_hidden_state,
            "decoder_hidden_state": self.get_decoder_last_hidden_state,

        }

    def __call__(self, nmt_in, nmt_out):
        result = {

        }
        for feature_name, layers in self.features.items():
            for layer in layers:
                feature, mask = self.functions[feature_name](nmt_in, nmt_out, layer)

                name = "{}_{}".format(feature_name, layer)
                result[name] = feature
                result["{}_mask".format(name)] = mask

        # result = {feature_name: self.functions[feature_name](nmt_in, nmt_out, layer) for feature_name, layer in self.features.items()}
        return result

    def get_encoder_last_hidden_state(self, nmt_in, nmt_out, layer):

        return (nmt_out["encoder_hidden_states"][layer], ~nmt_in["attention_mask"].bool())

    def get_decoder_last_hidden_state(self, nmt_in, nmt_out, layer):
        attention_mask_decoder = (self.padding_id != nmt_in["labels"]).long()
        return (nmt_out["decoder_hidden_states"][layer], ~attention_mask_decoder.bool())

    def get_decoder_last_hidden_attention(self, nmt_in, nmt_out):
        # Should check out how to do this. It is a bit confussing
        raise NotImplementedError()

    def get_feature_names(self):
        feature_names = []
        for feature_name, layers in self.features.items():
            for layer in layers:
                feature_names.append("{}_{}".format(feature_name, layer))
        return feature_names


def preprocess_function_pooled(self, batch):
    '''
    Function that preprocesses the dataset
    :param batch:
    :return:
    '''
    sources = batch["source"]
    hypotheses = batch["hypotheses"]
    with torch.no_grad():
        features = self.get_features_batch(sources, hypotheses)

    features = {k: v.cpu().numpy() for k, v in features.items()}

    return features


def remove_padding(layer, padding):
    '''
    Removes the vectors that are not attended to (entries where attention = 0) and returns the numpy array
    :param layer:
    :param attention:
    :return:
    '''
    layer = layer.cpu().numpy()
    attention = (~padding).cpu().numpy()

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
    sources = batch["source"]
    hypotheses = batch["hypotheses"]
    with torch.no_grad():
        features = self.get_features_batch(sources, hypotheses)

    feature_names = self.feature_names

    # Remove the padding
    result = {}
    for feature_name in feature_names:
        result[feature_name] = remove_padding(features[feature_name], features[feature_name + "_mask"])

    return result


preprocess_functions = {
    'pooled': preprocess_function_pooled,
    'full': preprocess_function_full,
}
