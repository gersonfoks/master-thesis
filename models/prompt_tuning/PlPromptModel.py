from torch import nn
from torch.nn import MSELoss

from custom_loss.GaussianMixtureLoss import GaussianMixtureLoss
from models.pl_predictive.PLBasePredictiveModel import PLBasePredictiveModel

from transformers import DataCollatorForSeq2Seq
import pytorch_lightning as pl
import torch

class PLPromptModel(pl.LightningModule):

    def __init__(self, nmt_model, tokenizer, head, encoder_prompt_embedding, decoder_prompt_embedding, initialize_optimizer, padding_id=-100,

                 device="cuda", n_mixtures=2):
        super().__init__()

        self.n_mixtures = n_mixtures

        self.initialize_optimizer = initialize_optimizer


        self.criterion = MSELoss()

        self.head = head.to(device)

        self.device_name = device
        self.nmt_model = nmt_model.to(self.device_name)
        self.nmt_model.requires_grad = False
        self.tokenizer = tokenizer

        self.encoder_prompt_embedding = encoder_prompt_embedding
        self.decoder_prompt_embedding = decoder_prompt_embedding

        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                                    padding=True, return_tensors="pt")

        self.n_prompts = encoder_prompt_embedding.shape[0]

        self.log_vars = ["loss"]



        self.nmt_model.train()





    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):

        # First we forward the input ids to the embedding and prepend the learned prompt_embedding. Also we update the attention_mask
        input_embedding, attention_mask = self.get_input_embeddings(input_ids, attention_mask)

        decoder_inputs_embeds , decoder_attention_mask = self.get_decoder_inputs_embeds(decoder_input_ids)

        nmt_out = self.nmt_model.forward(inputs_embeds=input_embedding, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask,
                                         decoder_inputs_embeds =decoder_inputs_embeds , output_hidden_states=True)

        features = nmt_out["decoder_hidden_states"][-1][:, 0] # Use the first token of the last hidden state

        predicted_utilities = self.head.forward(features)

        return predicted_utilities


    def get_input_embeddings(self, input_ids, attention_mask):

        input_embed = self.nmt_model.model.encoder.embed_tokens(input_ids)


        # Append
        prompt_embedding = self.encoder_prompt_embedding.unsqueeze(0).repeat(input_ids.shape[0], 1, 1) # Repeat for the batch
        input_embed = torch.cat([prompt_embedding, input_embed], 1)

        shape = attention_mask.shape

        one_vector = torch.ones((shape[0], self.n_prompts)).to(self.device)

        attention_mask = torch.cat([one_vector, attention_mask], 1)


        return input_embed, attention_mask

    def get_decoder_inputs_embeds(self, decoder_input_ids, ):



        input_embed = self.nmt_model.model.decoder.embed_tokens(decoder_input_ids)

        prompt_embedding = self.decoder_prompt_embedding.unsqueeze(0).repeat(decoder_input_ids.shape[0], 1,
                                                                             1)  # Repeat for the batch

        embed = torch.cat([prompt_embedding, input_embed], 1)

        shape = decoder_input_ids.shape

        one_vector = torch.ones((shape[0], self.n_prompts)).to(self.device)
        attention_mask = decoder_input_ids != 58100
        attention_mask[:, 0] = True # First padding shouldn't be ignored

        attention_mask = torch.cat([one_vector, attention_mask], 1)

        return embed, attention_mask



    def batch_to_out(self, batch):


        x, (sources, targets), utilities = batch


        x = {k: v.to("cuda") for k, v in x.items()}

        predicted_utility = self.forward(**x).flatten()


        utilities = utilities.to("cuda")


        loss = self.criterion(predicted_utility, utilities)

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
        return self.initialize_optimizer(list(self.head.parameters()) + [self.encoder_prompt_embedding, self.decoder_prompt_embedding] ) #

