import torch
from torch import nn

import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence


class EmbeddingLayer(nn.Module):
    '''
    Embedding layers maps the source, hypothesis and references to embedding space
    It returns a dictionary with the embeddings
    '''


    def __init__(self, source_embedding_layer, target_embedding_layer):
        super().__init__()
        self.source_embedding_layer = source_embedding_layer
        self.target_embedding_layer = target_embedding_layer


    def forward(self, tokenized_sources, tokenized_hypotheses, tokenized_references):

        embedded_sources = self.source_embedding_layer(tokenized_sources["input_ids"])
        embedded_hypotheses = self.target_embedding_layer(tokenized_hypotheses["input_ids"])


        embedded_references = []
        for ref in tokenized_references:
            temp = self.target_embedding_layer(ref["input_ids"])
            embedded_references.append(temp)



        return {"embedded_sources": embedded_sources,
                "embedded_hypotheses": embedded_hypotheses,
                "embedded_references": embedded_references,
                "attention_sources": ~tokenized_sources["attention_mask"].bool(),
                "attention_hypotheses": ~tokenized_hypotheses["attention_mask"].bool(),
                "attention_references": [~ref["attention_mask"].bool() for ref in tokenized_references],
                }



class LastHiddenStateEmbedding(nn.Module):

    def __init__(self, nmt_model):
        super().__init__()

        self.nmt_model = nmt_model
        nmt_model.eval()

        self.cache = {}

    def forward(self,  tokenized_sources, tokenized_hypotheses, tokenized_references, hypotheses_ids, reference_ids):

        (embedded_sources, embedded_hypotheses) = self.get_hidden_states(tokenized_sources["input_ids"], tokenized_sources["attention_mask"],
                                                              tokenized_hypotheses["input_ids"],
                                                              tokenized_hypotheses["attention_mask"], hypotheses_ids
                                                              )

        embedded_references = []
        for ref, ref_id in zip(tokenized_references, reference_ids):
            (_, ref_out) =  self.get_hidden_states(tokenized_sources["input_ids"], tokenized_sources["attention_mask"],
                                                              ref["input_ids"],
                                                              ref["attention_mask"],
                                                              )
            embedded_references.append(ref_out)

        return {"embedded_sources": embedded_sources,
                "embedded_hypotheses": embedded_hypotheses,
                "embedded_references": embedded_references,
                "attention_sources": ~tokenized_sources["attention_mask"].bool(),
                "attention_hypotheses": ~tokenized_hypotheses["attention_mask"].bool(),
                "attention_references": [~ref["attention_mask"].bool() for ref in tokenized_references],
                }

    @torch.no_grad()
    def get_hidden_states(self, source_ids, attention_mask_source, targets, target_attentions, ids):

        # First check which is in the cache in which aren't

        encoder_state = []
        decoder_state = []

        source_forward = []
        att_forward = []
        t_forward = []
        t_att_forward = []

        for c, id in enumerate(ids):
            if id in self.cache:
                temp_enc, temp_dec = self.cache[id]
                encoder_state.append(temp_enc)
                decoder_state.append(temp_dec)
            else:
                s_ids = source_ids[c]
                att_mask = attention_mask_source[c]
                t_id = targets[c]
                t_att = target_attentions[c]

                source_forward.append(s_ids)
                att_forward.append(att_mask)
                t_forward.append(t_id)
                t_att_forward.append(t_att)









        nmt_out = self.nmt_model.forward(input_ids=source_ids, attention_mask=attention_mask_source,
                                         decoder_attention_mask=targets,
                                         decoder_input_ids=target_attentions, output_hidden_states=True,
                                         output_attentions=False)
        encoder_state = nmt_out["encoder_last_hidden_state"]
        decoder_state = nmt_out["decoder_hidden_states"][-1]


        return encoder_state, decoder_state

    # @torch.no_grad()
    # def get_hidden_states_with_cache(self, source_ids, attention_mask_source, targets, target_attentions, ids):
    #
    #     # First check which is in the cache in which aren't
    #
    #     encoder_state = []
    #     decoder_state = []
    #
    #     source_forward = []
    #     att_forward = []
    #     t_forward = []
    #     t_att_forward = []
    #
    #     for c, id in enumerate(ids):
    #         if id in self.cache:
    #             temp_enc, temp_dec = self.cache[id]
    #             encoder_state.append(temp_enc)
    #             decoder_state.append(temp_dec)
    #         else:
    #             s_ids = source_ids[c]
    #             att_mask = attention_mask_source[c]
    #             t_id = targets[c]
    #             t_att = target_attentions[c]
    #
    #             source_forward.append(s_ids)
    #             att_forward.append(att_mask)
    #             t_forward.append(t_id)
    #             t_att_forward.append(t_att)
    #
    #     nmt_out = self.nmt_model.forward(input_ids=source_ids, attention_mask=attention_mask_source,
    #                                      decoder_attention_mask=targets,
    #                                      decoder_input_ids=target_attentions, output_hidden_states=True,
    #                                      output_attentions=False)
    #     encoder_state = nmt_out["encoder_last_hidden_state"]
    #     decoder_state = nmt_out["decoder_hidden_states"][-1]
    #
    #     return encoder_state, decoder_state



class AttentionFeatureExtractionLayer(nn.Module):

    def __init__(self, source_attention, target_attention, source_pool, target_pool, n_references=3):
        super().__init__()
        self.source_attention = source_attention
        self.target_attention = target_attention
        self.source_pool = source_pool
        self.target_pool = target_pool
        self.n_references = n_references


    def forward(self, embedded_sources, embedded_hypotheses, embedded_references, attention_sources, attention_hypotheses, attention_references):



        source_features, _ = self.source_attention(embedded_hypotheses, embedded_sources, embedded_sources, key_padding_mask=attention_sources, )

        source_features = self.source_pool(source_features, attention_hypotheses)

        hypotheses_features, _ = self.target_attention(embedded_hypotheses, embedded_hypotheses, embedded_hypotheses,
                                                key_padding_mask=attention_hypotheses, )
        hypotheses_features = self.target_pool(hypotheses_features, attention_hypotheses)

        # Here we need to repeat the embedding hypotheses such that it follows the

        reference_features = []
        for ref, att_ref in zip(embedded_references, attention_references):

            temp, _ = self.target_attention(embedded_hypotheses, ref, ref,
                                                    key_padding_mask=att_ref, )
            temp = self.target_pool(temp, attention_hypotheses)

            reference_features.append(temp)

        diff_feat = []
        for ref_feat in reference_features:

            temp = torch.abs(hypotheses_features - ref_feat)
            diff_feat.append(temp)


        # Lastly we concatonate:

        features = torch.concat([source_features, hypotheses_features] + reference_features + diff_feat, dim=1)

        return features


class PoolFeatureExtractionLayer(nn.Module):

    def __init__(self,  source_pool, target_pool, n_references=3):
        super().__init__()
        self.source_pool = source_pool
        self.target_pool = target_pool
        self.n_references = n_references


    def forward(self, embedded_sources, embedded_hypotheses, embedded_references, attention_sources, attention_hypotheses, attention_references):


        source_features = self.source_pool(embedded_sources, attention_sources)

        hypotheses_features = self.target_pool(embedded_hypotheses, attention_hypotheses)

        # Here we need to repeat the embedding hypotheses such that it follows the

        reference_features = []
        for ref, att_ref in zip(embedded_references, attention_references):

            temp = self.target_pool(ref, att_ref)

            reference_features.append(temp)

        # Also calculate the differences

        diff_feat = []
        for ref_feat in reference_features:
            temp = torch.abs(hypotheses_features - ref_feat)
            diff_feat.append(temp)





        # Lastly we concatonate:

        features = torch.concat([source_features, hypotheses_features] + reference_features + diff_feat, dim=1)

        return features



class LSTMFeatureExtractionLayer(nn.Module):

    def __init__(self, source_lstm, target_lstm, n_references=3):
        super().__init__()
        self.source_lstm = source_lstm
        self.target_lstm = target_lstm



    def forward(self, embedded_sources, embedded_hypotheses, embedded_references, attention_sources, attention_hypotheses, attention_references):

        # First we create pack sequences

        packed_embedded_sources = self.pack_sequence(embedded_sources, attention_sources)
        packed_embedded_hypotheses = self.pack_sequence(embedded_hypotheses, attention_hypotheses)




        _, (source_features, _) = self.source_lstm(packed_embedded_sources)
        _, (hypotheses_features, _) = self.source_lstm(packed_embedded_hypotheses)

        # Here we need to repeat the embedding hypotheses such that it follows the

        reference_features = []
        for ref, att_ref in zip(embedded_references, attention_references):
            packed_ref = self.pack_sequence(ref, att_ref)

            _, (temp, _) = self.target_lstm(packed_ref)

            reference_features.append(temp)

        # Calculate the difference and pointwise product:

        diff_feat = []
        for ref_feat in reference_features:
            diff_temp = torch.abs(hypotheses_features - ref_feat)
            diff_feat.append(diff_temp)







        # Lastly we concatonate:

        features = torch.concat([source_features, hypotheses_features] + reference_features + diff_feat, dim=-1)

        return features


    def pack_sequence(self, sequence, attention_mask):
        attention_mask = attention_mask.cpu().numpy()
        lengths = torch.tensor(np.argmax(attention_mask, axis=-1)) +1

        # The longest sequence has argmax 0 thus need to set that index to the max length.
        lengths = lengths + (lengths == 1) * (sequence.shape[1]-1)

        packed_sequence = pack_padded_sequence(sequence, lengths=lengths, batch_first=True, enforce_sorted=False)


        return packed_sequence

class PoolLayer(nn.Module):

    def __init__(self, hidden_dim, n_heads, n_queries=1, dropout=0.0, batch_first=True):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout,
                                                 batch_first=batch_first)

        self.query = nn.Parameter(torch.rand(size=(1, n_queries, 512)))

        torch.nn.init.normal_(self.query)


    def forward(self, x, attention_mask):
        query = self.query.repeat( x.shape[0], 1, 1) # Repeat the query enough times
        pooled, _ = self.attention(query, x, x, key_padding_mask=attention_mask)
        return pooled.squeeze(dim=1)






class TokenizerLayer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def forward(self, sources, hypotheses, references):
        tokenized_sources = self.tokenizer(sources, truncation=True, padding=True, return_tensors="pt", ).to("cuda")
        # input_ids = tokenized_sources["input_ids"]
        # attention_mask = tokenized_sources["attention_mask"]
        # Setup the tokenizer for targets

        tokenized_references= []

        with self.tokenizer.as_target_tokenizer():
            tokenized_hypotheses = self.tokenizer(hypotheses, truncation=True, padding=True, return_tensors="pt",).to(
                "cuda")
            for ref in references:

                temp = self.tokenizer(ref, truncation=True, padding=True, return_tensors="pt", ).to(
                "cuda")
                tokenized_references.append(temp)


        return tokenized_sources, tokenized_hypotheses, tokenized_references