

from datetime import datetime

import pytorch_lightning as pl
import torch
from torch.nn import MSELoss


class ReferenceModel(pl.LightningModule):

    def __init__(self, tokenizer_layer, embedding_layers, feature_extraction_layers, final_layers, initialize_optimizer, padding_id=-100, device="cuda", ):
        super().__init__()
        self.device_name = device

        self.tokenizer_layer = tokenizer_layer
        self.embedding_layers = embedding_layers
        self.feature_extraction_layers = feature_extraction_layers
        self.final_layers = final_layers





        self.padding_id = padding_id

        # Need to specify in settings below
        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, sources, hypotheses, references, hyp_ids, ref_ids):
        tokenized_sources, tokenized_hypotheses, tokenized_references = self.tokenizer_layer.forward(sources, hypotheses, references)

        embeddings = self.embedding_layers.forward(tokenized_sources, tokenized_hypotheses, tokenized_references, hyp_ids, ref_ids )

        features = self.feature_extraction_layers(**embeddings)


        predicted_scores = self.final_layers(features)


        return predicted_scores

    def get_predicted_risk(self, sources, hypotheses, references):
        raise NotImplementedError()

    def batch_to_out(self, batch):

        sources, hypotheses, references, scores, hyp_ids, ref_ids = batch
        predicted_scores = self.forward(sources, hypotheses, references, hyp_ids, ref_ids).flatten()

        loss = self.criterion(predicted_scores, scores.to(self.device))

        return {"loss": loss}




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

    @torch.no_grad()
    def predict(self, sources, hypotheses, references):
        '''
        Predicts the bayes risk for the source targets pairs
        :param sources:
        :param targets:
        :return:
        '''

        pass

    def configure_optimizers(self):

        return self.initialize_optimizer(list(self.embedding_layers.parameters()) + list(self.feature_extraction_layers.parameters()) + list(self.final_layers.parameters()))

