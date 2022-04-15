from datetime import datetime

import pytorch_lightning as pl
from torch import nn
from torch.nn import MSELoss, ReLU, SiLU
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


class BIListmModel(nn.Module):

    def __init__(self, vocab_size, embedding_size=512, hidden_size=512):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.hidden_size = 512

        self.dropout = 0.0

        self.num_layers = 2
        self.bidirectional = 1
        self.batch_first = True
        self.lstm_layer = torch.nn.LSTM(embedding_size, hidden_size=hidden_size, num_layers=self.num_layers, bidirectional=bool(self.bidirectional), batch_first=True, dropout=self.dropout)

        self.multiply_factor = self.num_layers  * 2 if self.bidirectional else self.num_layers

        self.linear_layers_sizes = [
            2048,
            1024,
            512,
            256,
            128,
            1
        ]

        self.linear_layers = self.create_linear_layers()

    # def element_wise_apply(self, fn, packed_sequence):
    #     # from: https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/3
    #     """applies a pointwise function fn to each element in packed_sequence"""
    #     return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

    def element_wise_apply(self, fn, packed_sequence):
        # from: https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/3
        """applies a pointwise function fn to each element in packed_sequence"""
        # embeddings = fn(packed_sequence.data)
        # print(embeddings)
        # print(embeddings.shape)
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes, sorted_indices=packed_sequence.sorted_indices, unsorted_indices=packed_sequence.unsorted_indices)

    def create_linear_layers(self):
        activation_function = ReLU()

        layers = []
        # Add all the layers except the last one
        for layer_in, layer_out in zip(self.linear_layers_sizes[:-2], self.linear_layers_sizes[1:-1]):
            layers.append(nn.Linear(layer_in, layer_out))
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))

            layers.append(activation_function)



        # Add the last one
        layers.append(nn.Linear(self.linear_layers_sizes[-2], self.linear_layers_sizes[-1]))
        # Also add the last activation function

        layers.append(nn.Sigmoid())# Append a tanh layer (to make sure to bound the output values)

        return nn.Sequential(*layers)


    def forward(self, x):
        x_embedded = self.element_wise_apply(self.embedding, x)

        output, (h_n, c_n) = self.lstm_layer(x_embedded)

        h_n = h_n.permute(1,0, 2)
        h_n = h_n.reshape(-1, self.hidden_size * self.multiply_factor)


        #linear_layer_input = torch.cat([h_n, c_n], dim=-1)

        predicted_score = self.linear_layers(h_n)


        return predicted_score


class PlLSTMModel(pl.LightningModule):

    def __init__(self, model, lr=0.001):
        super().__init__()


        # Need to specify in settings below
        self.criterion = MSELoss()

        self.log_vars = {
            "loss"
        }

        self.lr = lr


        self.model = model.to("cuda")




    def forward(self, input_ids):
        return self.model.forward(input_ids)

    def get_predicted_risk(self, input_ids, attention_mask, labels, decoder_input_ids):
        raise NotImplementedError()

    def batch_to_out(self, batch):

        input_ids, score = batch
        input_ids = input_ids.to("cuda")
        predicted_score = self.forward(input_ids).flatten()


        score = score.to("cuda")

        loss = self.criterion(predicted_score, score)
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


    def configure_optimizers(self):

        # lr_config = {
        #     "optimizer": torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0),
        #
        # }

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)
        #
        # lr_config = {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #
        #         "scheduler": ReduceLROnPlateau(optimizer, "min", patience=2, ),
        #
        #         "monitor": 'train_loss'
        #     }
        #
        # }

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)

        lr_config = {
            "optimizer": optimizer,
            "lr_scheduler": {

                "scheduler": StepLR(optimizer, step_size=5, gamma=0.7, verbose=True ),

                "monitor": 'train_loss'
            }

        }

        return lr_config

