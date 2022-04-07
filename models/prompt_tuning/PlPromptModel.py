from torch.nn import MSELoss
import numpy as np
from transformers import DataCollatorForSeq2Seq
import pytorch_lightning as pl
import torch


class PLPromptModel(pl.LightningModule):

    def __init__(self, nmt_model, tokenizer, head, encoder_prompt_embedding, start_decoder_prompt_embedding,
                 end_decoder_prompt_embedding, initialize_optimizer, padding_id=-100,

                 device="cuda", ):
        super().__init__()

        self.initialize_optimizer = initialize_optimizer

        self.criterion = MSELoss()

        self.head = head.to(device)

        self.device_name = device
        self.nmt_model = nmt_model.to(self.device_name)
        self.nmt_model.requires_grad = False
        self.tokenizer = tokenizer

        self.encoder_prompt_embedding = encoder_prompt_embedding
        self.start_decoder_prompt_embedding = start_decoder_prompt_embedding
        self.end_decoder_prompt_embedding = end_decoder_prompt_embedding

        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                                    padding=True, return_tensors="pt")

        self.log_vars = ["loss"]

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):

        # First we forward the input ids to the embedding and prepend the learned prompt_embedding. Also we update the attention_mask

        input_embedding, attention_mask = self.get_input_embeddings(input_ids, attention_mask)
        if type(self.start_decoder_prompt_embedding) != type(None):
            decoder_inputs_embeds, decoder_attention_mask, last_state_indices = self.get_decoder_inputs_embeds(
                decoder_input_ids)
        else:
            decoder_inputs_embeds, decoder_attention_mask, last_state_indices = self.get_decoder_inputs_embeds_2(
                decoder_input_ids)

        nmt_out = self.nmt_model.forward(inputs_embeds=input_embedding, attention_mask=attention_mask,
                                         decoder_attention_mask=decoder_attention_mask,
                                         decoder_inputs_embeds=decoder_inputs_embeds, output_hidden_states=True,
                                         output_attentions=True)

        features = []

        for b, i in enumerate(last_state_indices):
            features.append(nmt_out["decoder_hidden_states"][-1][b][i])
        features = torch.stack(features)

        # attentions = nmt_out["decoder_attentions"][-1][:, :, 0]  # Use the first token of the last hidden state
        predicted_utilities = self.head.forward(features)

        return predicted_utilities

    def get_input_embeddings(self, input_ids, attention_mask):
        enc = self.nmt_model.model.encoder
        if type(self.encoder_prompt_embedding) == type(None):
            return enc.embed_tokens(input_ids) * enc.embed_scale, attention_mask

        input_embed = enc.embed_tokens(input_ids) * enc.embed_scale

        prompt_embedding = self.encoder_prompt_embedding.unsqueeze(0).repeat(input_ids.shape[0], 1,
                                                                             1)  # Repeat for the batch
        input_embed = torch.cat([prompt_embedding, input_embed], 1)

        shape = attention_mask.shape

        one_vector = torch.ones((shape[0], self.encoder_prompt_embedding.shape[0])).to(self.device)

        attention_mask = torch.cat([one_vector, attention_mask], 1)

        return input_embed, attention_mask

    def get_decoder_inputs_embeds(self, decoder_input_ids, ):

        # First we append n_prompts amount of padding to the lists (this to create space for the end embedding)

        new_decoder_input_ids = torch.cat(
            [torch.ones(decoder_input_ids.shape[0], self.start_decoder_prompt_embedding.shape[0]).to("cuda") * 58100,
             decoder_input_ids,
             torch.ones(decoder_input_ids.shape[0], self.end_decoder_prompt_embedding.shape[0]).to("cuda") * 58100, ],
            1).int()
        dec = self.nmt_model.model.decoder
        embed = dec.embed_tokens(new_decoder_input_ids) * dec.embed_scale

        # Prepend the start embedding
        start_prompt_embedding = self.start_decoder_prompt_embedding.unsqueeze(0).repeat(decoder_input_ids.shape[0], 1,
                                                                                         1)  # Repeat for the batch

        n_start_prompts = self.start_decoder_prompt_embedding.shape[0]
        n_end_prompts = self.end_decoder_prompt_embedding.shape[0]
        embed[:, :n_start_prompts] = start_prompt_embedding

        shape = decoder_input_ids.shape

        # Next we get the places where we have padding and insert the end embedding
        paddings_indices = (new_decoder_input_ids == 58100).nonzero(as_tuple=True)

        batch_index = paddings_indices[0].cpu().numpy()

        resulting_index = []

        batch_size = embed.shape[0]
        paddings_indices = (new_decoder_input_ids == 58100).nonzero()
        for i in range(batch_size):
            # Ignore the first n_prompts + 1
            start_index = int(np.where(batch_index == i)[0][0])

            resulting_index.append(paddings_indices[
                                   start_index + n_start_prompts + 1: start_index + n_start_prompts + n_end_prompts + 1].cpu().tolist())

            # Then we update
        # end_prompt_indices = torch.tensor(resulting_index).to("cuda")
        # print(end_prompt_indices)
        last_state_indices = []
        for indices in resulting_index:
            last_state_indices.append(indices[-1][-1])
            for c, i in enumerate(indices):
                embed[i[0], i[1]] = self.end_decoder_prompt_embedding[c]

        one_vector = torch.ones((shape[0], n_start_prompts + n_end_prompts)).to(self.device)
        attention_mask = decoder_input_ids != 58100
        attention_mask[:, 0] = True  # First padding shouldn't be ignored

        attention_mask = torch.cat([one_vector, attention_mask, ], 1)

        return embed, attention_mask, last_state_indices

    def get_decoder_inputs_embeds_2(self, decoder_input_ids, ):

        # First we append n_prompts amount of padding to the lists (this to create space for the end embedding)
        n_end_prompts = self.end_decoder_prompt_embedding.shape[0]
        new_decoder_input_ids = torch.cat(
            [decoder_input_ids, torch.ones(decoder_input_ids.shape[0], n_end_prompts).to("cuda") * 58100, ], 1).int()

        embed = self.nmt_model.model.decoder.embed_tokens(new_decoder_input_ids)

        shape = decoder_input_ids.shape

        # Next we get the places where we have padding and insert the end embedding
        paddings_indices = (new_decoder_input_ids == 58100).nonzero(as_tuple=True)

        batch_index = paddings_indices[0].cpu().numpy()

        resulting_index = []

        batch_size = embed.shape[0]
        paddings_indices = (new_decoder_input_ids == 58100).nonzero()
        for i in range(batch_size):
            # Ignore the first n_prompts + 1
            start_index = int(np.where(batch_index == i)[0][0])

            resulting_index.append(paddings_indices[start_index + 1: start_index + n_end_prompts + 1].cpu().tolist())

            # Then we update
        # end_prompt_indices = torch.tensor(resulting_index).to("cuda")
        # print(end_prompt_indices)
        last_state_indices = []
        for indices in resulting_index:
            last_state_indices.append(indices[-1][-1])
            for c, i in enumerate(indices):
                embed[i[0], i[1]] = self.end_decoder_prompt_embedding[c]

        one_vector = torch.ones((shape[0], n_end_prompts)).to(self.device)
        attention_mask = decoder_input_ids != 58100
        attention_mask[:, 0] = True  # First padding shouldn't be ignored

        attention_mask = torch.cat([one_vector, attention_mask, ], 1)

        return embed, attention_mask, last_state_indices

    def batch_to_out(self, batch):

        x, (sources, targets), utilities = batch

        x = {k: v.to("cuda") for k, v in x.items()}

        predicted_utility = self.forward(**x).flatten()

        # print(self.start_decoder_prompt_embedding.grad)
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
        prompt_embeddings = [self.end_decoder_prompt_embedding]
        if type(self.start_decoder_prompt_embedding) != type(None):
            prompt_embeddings.append(self.start_decoder_prompt_embedding)
        if type(self.encoder_prompt_embedding) != type(None):
            prompt_embeddings.append(self.encoder_prompt_embedding)

        return self.initialize_optimizer(list(self.head.parameters()) + prompt_embeddings)  #


class PLFinetuneModel(pl.LightningModule):

    def __init__(self, nmt_model, tokenizer, head, initialize_optimizer, padding_id=-100,

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

        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                                    padding=True, return_tensors="pt")

        self.log_vars = ["loss"]

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):

        nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                         decoder_input_ids=decoder_input_ids, output_hidden_states=True)

        features = nmt_out["decoder_hidden_states"][-1][:, 0]  # Use the first token of the last hidden state

        predicted_utilities = self.head.forward(features)

        return predicted_utilities

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
        return self.initialize_optimizer(list(self.head.parameters()) + list(self.nmt_model.parameters()))  #
