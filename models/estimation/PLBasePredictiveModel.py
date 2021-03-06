from datetime import datetime

import pytorch_lightning as pl
import torch


class PLBasePredictiveModel(pl.LightningModule):

    def __init__(self, nmt_model, tokenizer, head, initialize_optimizer, padding_id=-100, device="cuda", ):
        super().__init__()
        self.device_name = device
        self.nmt_model = nmt_model.to(self.device_name)
        self.nmt_model.requires_grad = False
        self.tokenizer = tokenizer
        self.nmt_model.eval()  # Make sure we set the nmt_model to evaluate
        # Initialize the predictive layers

        self.head = head
        self.head.to(self.device_name)


        self.padding_id = padding_id

        # Need to specify in settings below
        self.criterion = None

        self.log_vars = {
            "loss"
        }

        self.initialize_optimizer = initialize_optimizer

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):
        raise NotImplementedError()

    def get_predicted_risk(self, input_ids, attention_mask, labels, decoder_input_ids):
        raise NotImplementedError()

    def batch_to_out(self, batch):
        raise NotImplementedError()

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
    def predict(self, sources, targets):
        '''
        Predicts the bayes risk for the source targets pairs
        :param sources:
        :param targets:
        :return:
        '''

        tokenized_sources = self.tokenizer(sources, truncation=True, padding=True, return_tensors="pt", ).to("cuda")
        input_ids = tokenized_sources["input_ids"]
        attention_mask = tokenized_sources["attention_mask"]
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, truncation=True, padding=True, return_tensors="pt")["input_ids"].to("cuda")

        # Prepadding
        padding = torch.ones((labels.shape[0], 1)).to("cuda") * self.tokenizer.pad_token_id

        decoder_input_ids = torch.cat([padding, labels], dim=-1)[:, :-1].long()

        # labels
        is_padding = labels == self.tokenizer.pad_token_id

        labels = labels * ~is_padding + self.padding_id * is_padding

        model_out = self.forward(input_ids, attention_mask, labels, decoder_input_ids)

        return model_out

    def configure_optimizers(self):
        return self.initialize_optimizer(self.head.parameters())

