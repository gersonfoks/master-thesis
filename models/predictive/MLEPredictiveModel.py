import pytorch_lightning as pl
import torch
from torch import nn

from models.predictive.BasePredictiveModel import BasePredictiveModel
from models.predictive.GaussianPredictiveModel import avg_pooling
from models.predictive.pool_utils import max_pooling


class MLEPredictiveModel(BasePredictiveModel):

    def __init__(self, nmt_model, tokenizer, predictive_layers, padding_id=-100, lr=0.00005, weight_decay=1e-5,
                 device="cuda", avg_pool=True, max_pool=True):
        super().__init__(nmt_model, tokenizer, predictive_layers, padding_id=padding_id, lr=lr,
                         weight_decay=weight_decay,
                         device="cuda", )

        self.avg_pool = avg_pool
        self.max_pool = max_pool

        if not self.avg_pool and not self.max_pool:
            raise ValueError("We should have at least one type of pooling!")

        self.criterion = nn.MSELoss()

    def get_features(self, input_ids, attention_mask, labels, decoder_input_ids):
        nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                         decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                         output_attentions=True)
        encoder_last_hidden_state = nmt_out["encoder_last_hidden_state"]
        decoder_last_hidden_state = nmt_out["decoder_hidden_states"][-1]

        # Next perform average pooling
        # first apply attention_mask to encoder_last_hidden_state
        pooled_states = []
        attention_mask_decoder = (self.padding_id != labels).long()
        if self.avg_pool:

            avg_encoder_hidden_state = avg_pooling(encoder_last_hidden_state, attention_mask)

            avg_decoder_hidden_state = avg_pooling(decoder_last_hidden_state, attention_mask_decoder)

            pooled_states += [avg_encoder_hidden_state,avg_decoder_hidden_state ]

        if self.max_pool:
            max_encoder_hidden_state = max_pooling(encoder_last_hidden_state, attention_mask)

            max_decoder_hidden_state = max_pooling(decoder_last_hidden_state, attention_mask_decoder)

            pooled_states += [max_encoder_hidden_state, max_decoder_hidden_state]

        # Concat the two
        hidden_states_concat = torch.cat(pooled_states, dim=-1)

        return hidden_states_concat

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):
        features = self.get_features(input_ids, attention_mask, labels, decoder_input_ids)

        x = self.predictive_layers(features)

        return x

    def get_predicted_risk(self, input_ids, attention_mask, labels, decoder_input_ids):
        return self.forward(input_ids, attention_mask, labels, decoder_input_ids)

    def batch_to_out(self, batch):
        x, (sources, targets), utilities = batch

        x = {k: v.to("cuda") for k, v in x.items()}

        predicted_risk = self.forward(**x)

        utilities = utilities.to("cuda")
        predicted_risk = predicted_risk.flatten()

        loss = self.criterion(predicted_risk, utilities)

        return {"loss": loss}
