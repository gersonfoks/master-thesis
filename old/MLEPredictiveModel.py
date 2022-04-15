import pytorch_lightning as pl
import torch
from torch import nn

from models.predictive.BasePredictiveModel import BasePredictiveModel
from models.predictive.GaussianPredictiveModel import avg_pooling
from models.predictive.pool_utils import max_pooling
from transformers import DataCollatorForSeq2Seq


class MLEPredictiveModel(BasePredictiveModel):

    def __init__(self, nmt_model, tokenizer, predictive_layers, padding_id=-100, lr=0.0005, weight_decay=1e-5,
                 device="cuda", avg_pool=True, max_pool=True):
        super().__init__(nmt_model, tokenizer, predictive_layers, padding_id=padding_id, lr=lr,
                         weight_decay=weight_decay,
                         device="cuda", )

        self.avg_pool = avg_pool
        self.max_pool = max_pool

        if not self.avg_pool and not self.max_pool:
            raise ValueError("We should have at least one type of pooling!")

        self.criterion = nn.MSELoss()

        self.mode = "text"

        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                                    padding=True, return_tensors="pt")
        self.feature_names = ["avg_pool_encoder_hidden_state",
                              "max_pool_encoder_hidden_state",
                              "avg_pool_decoder_hidden_state",
                              "max_pool_decoder_hidden_state"
                              ]

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):

        features = self.get_features(input_ids, attention_mask, labels, decoder_input_ids)

        return self.forward_features(features)

    def forward_features(self, features):

        feature_list = [
            features[name] for name in self.feature_names
        ]
        cat_features = torch.cat(feature_list, dim=-1)

        x = self.predictive_model(cat_features)

        return x

    def get_predicted_risk(self, input_ids, attention_mask, labels, decoder_input_ids):
        return self.forward(input_ids, attention_mask, labels, decoder_input_ids)

    def batch_to_out(self, batch):

        if self.mode == "text":
            x, (sources, targets), utilities = batch

            x = {k: v.to("cuda") for k, v in x.items()}

            predicted_risk = self.forward(**x)

            utilities = utilities.to("cuda")
            predicted_risk = predicted_risk.flatten()

            loss = self.criterion(predicted_risk, utilities)

            return {"loss": loss}

        else:
            features, (sources, hypothesis), utilities = batch

            features = {k: v.to("cuda") for k, v in features.items()}
            predicted_risk = self.forward_features(features)

            utilities = utilities.to("cuda")
            predicted_risk = predicted_risk.flatten()

            loss = self.criterion(predicted_risk, utilities)

            return {"loss": loss}

    def get_features(self, input_ids, attention_mask, labels, decoder_input_ids):
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

    def get_features_batch(self, sources, hypothesis):

        model_inputs = self.tokenizer(sources, truncation=True, )
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(hypothesis, truncation=True, )

        model_inputs["labels"] = labels["input_ids"]

        x = [{"labels": l, "input_ids": i, "attention_mask": a} for (l, i, a) in
             zip(model_inputs["labels"], model_inputs["input_ids"], model_inputs["attention_mask"])]

        data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                               padding=True, return_tensors="pt")

        x_new = data_collator(x).to("cuda")
        features = self.get_features(**x_new)

        return features

    def set_mode(self, mode):
        possible_modes = ["text", "features"]
        if mode not in ["text", "features"]:
            raise ValueError("Mode is {} should be in {}".format(mode, possible_modes))
        self.mode = mode

    def preprocess_function(self, batch):
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
