import torch
from torch import nn
from torch.nn import MSELoss
import numpy as np
from custom_loss.GaussianMixtureLoss import GaussianMixtureLoss
from models.estimation.PLBasePredictiveModel import PLBasePredictiveModel

from transformers import DataCollatorForSeq2Seq

import pytorch_lightning as pl

from models.estimation.feature_functions import remove_padding


class ReferenceMSEPredictiveModel(pl.LightningModule):

    def __init__(self, nmt_model, tokenizer, head, feature_names, initialize_optimizer, padding_id=-100,
                 device="cuda", ):
        super().__init__()
        self.device_name = device
        self.nmt_model = nmt_model.to(self.device_name)
        self.nmt_model.requires_grad = False
        self.tokenizer = tokenizer
        self.nmt_model.eval()  # Make sure we set the nmt_model to evaluate
        # Initialize the predictive layers

        self.head = head
        self.head.to(self.device_name)

        self.padding_id_labels = padding_id
        self.padding_id = 58100

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

        self.criterion = MSELoss()

        self.mode = "text"

        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                                    padding=True, return_tensors="pt")
        self.feature_names = feature_names




        self.top_n = 3

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids, references_ids):
        features = self.get_features(input_ids, attention_mask, labels, decoder_input_ids, references_ids)

        return self.forward_features(features)

    def forward_features(self, features):

        out = self.head.forward(features)

        return out

    def get_predicted_risk(self, input_ids, attention_mask, labels, decoder_input_ids):
        raise NotImplementedError()

    def batch_to_out(self, batch):
        if self.mode == "text":
            x, (sources, hypothesis), utilities = batch

            x = {k: v.to("cuda") for k, v in x.items()}

            prediction = self.forward(**x)

        else:
            features, (sources, hypothesis), utilities = batch

            features = {k: v.to("cuda") for k, v in features.items()}
            prediction = self.forward_features(features)

        utilities = utilities.to("cuda")
        prediction = prediction.flatten()

        loss = self.criterion(prediction, utilities)

        return {"loss": loss}

    def get_features_batch(self, sources, hypothesis, top_n_hypotheses):
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
        reference_ids = []
        top_n_hypotheses = np.array(top_n_hypotheses)
        # Preprocess the reference hypotheses
        for i in range(self.top_n):

            hypotheses = top_n_hypotheses[:, i].tolist()

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(hypotheses, truncation=True, )

            model_inputs["labels"] = labels["input_ids"]
            x_temp = [{"labels": l, "input_ids": i, "attention_mask": a} for (l, i, a) in
             zip(model_inputs["labels"], model_inputs["input_ids"], model_inputs["attention_mask"])]

            reference_ids.append(data_collator(x_temp)["decoder_input_ids"].to("cuda"))



        x_new["references_ids"] = reference_ids

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

        sources = batch["source"]
        hypotheses = batch["hypotheses"]
        top_n_hypotheses = batch["top_n_hypotheses"]
        feature_names = self.feature_names

        with torch.no_grad():
            features = self.get_features_batch(sources, hypotheses, top_n_hypotheses)



        # Remove the padding
        result = {}
        for feature_name in feature_names:
            result[feature_name] = remove_padding(features[feature_name], features[feature_name + "_mask"])

        return result

    def get_features(self, input_ids, attention_mask, labels, decoder_input_ids, references_ids, ):

        nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                         decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                         output_attentions=True)

        # Create the mask for the references
        reference_nmt_out = {}
        for i, references in enumerate(references_ids):


            decoder_mask = (references != self.padding_id).bool()
            decoder_mask[:, 0] = True  # The first padding must be ignored

            temp_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask,
                                              decoder_attention_mask=decoder_mask,
                                              decoder_input_ids=references, output_hidden_states=True,
                                              output_attentions=True)
            reference_nmt_out['reference_{}'.format(i )] = temp_out['encoder_hidden_states'][-1]
            reference_nmt_out['reference_{}_mask'.format(i )] = ~decoder_mask

        attention_mask_decoder = (self.padding_id_labels != labels).long()

        return {
            'decoder_hidden_state_-1': nmt_out["decoder_hidden_states"][-1],
            'encoder_hidden_state_-1': nmt_out["encoder_hidden_states"][-1],
            'encoder_hidden_state_-1_mask': attention_mask,
            'decoder_hidden_state_-1_mask': attention_mask_decoder,
            **reference_nmt_out,
        }

    def training_step(self, batch, batch_idx):

        batch_out = self.batch_to_out(batch)


        loss = batch_out["loss"]

        for log_var in self.log_vars:
            self.log("train_{}".format(log_var), batch_out[log_var])



        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_out = self.batch_to_out(batch)


        for log_var in self.log_vars:
            self.log("val_{}".format(log_var), batch_out[log_var])


    def configure_optimizers(self):
        return self.initialize_optimizer(self.head.parameters())

