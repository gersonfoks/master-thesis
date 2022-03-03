import pytorch_lightning as pl
import torch
from torch import nn

from models.pl_predictive.PLBasePredictiveModel import PLBasePredictiveModel

from transformers import DataCollatorForSeq2Seq


def count_to_mean(counter):
    total_count = 0
    total = 0
    for value, c in counter.items():
        total += value * c
        total_count += c

    return total / total_count


class MSEPredictiveModel(PLBasePredictiveModel):

    def __init__(self, nmt_model, tokenizer, predictive_model, feature_names, initialize_optimizer, padding_id=-100,
                 lr=0.0005, weight_decay=1e-5,
                 device="cuda", ):
        super().__init__(nmt_model, tokenizer, predictive_model, initialize_optimizer, padding_id=padding_id,
                         device=device, )

        self.criterion = nn.MSELoss()

        self.mode = "text"

        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                                    padding=True, return_tensors="pt")
        self.feature_names = feature_names

        self.predictive_model = self.predictive_model.to(device)

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):

        features = self.get_features(input_ids, attention_mask, labels, decoder_input_ids)

        return self.forward_features(features)

    def forward_features(self, features):
        return self.predictive_model.forward(features)

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
        pass

    def utility_transform(self, utilities):
        '''
        Transform the utilities to something we can to work with
        :param utilities:
        :return:
        '''
        return count_to_mean(utilities)

    def get_features(self, input_ids, attention_mask, labels, decoder_input_ids):
        raise NotImplementedError()
