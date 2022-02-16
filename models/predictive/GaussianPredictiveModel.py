import pytorch_lightning as pl
import torch
from torch import nn

from models.predictive.pool_utils import avg_pooling


class GaussianPredictiveModel(pl.LightningModule):

    def __init__(self, nmt_model, tokenizer, padding_id=-100, device="cuda", lr=0.00005, weight_decay=1e-5):
        super().__init__()
        self.nmt_model = nmt_model.to("cuda")
        self.nmt_model.requires_grad = False
        self.tokenizer = tokenizer
        self.nmt_model.eval()  # Make sure we set the nmt_model to evaluate
        # Initialize the predictive layers

        self.linear_layers = nn.Sequential(nn.Linear(512 * 2, 512), torch.nn.SiLU(), nn.Dropout(p=0.25),
                                           nn.Linear(512, 1))
        self.linear_layers.to("cuda")
        self.softplus = nn.Softplus()
        self.padding_id = padding_id

        self.lr = lr
        self.weight_decay = weight_decay

        # We include the constant to get a good idea on how well we are doing (to make sure we have tot 0 is the optimal value
        self.criterion = nn.GaussianNLLLoss(full=True)

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):
        nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                         decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                         output_attentions=True)
        encoder_last_hidden_state = nmt_out["encoder_last_hidden_state"]
        decoder_last_hidden_state = nmt_out["decoder_hidden_states"][-1]

        # Next perform average pooling
        # first apply attention_mask to encoder_last_hidden_state

        avg_encoder_hidden_state = avg_pooling(encoder_last_hidden_state, attention_mask)
        # Then devide

        attention_mask_decoder = (self.padding_id != labels).long()
        avg_decoder_hidden_state = avg_pooling(decoder_last_hidden_state, attention_mask_decoder)

        # Concat the two
        hidden_states_concat = torch.cat([avg_encoder_hidden_state, avg_decoder_hidden_state], dim=-1)

        x = self.linear_layers(hidden_states_concat)
        # Next we apply softplus to get the standard deviation
        avg = x[:, 0:1]
        std = x[:, 1:]

        std = self.softplus(std)
        return avg, std

    def training_step(self, batch, batch_idx):
        x, (sources, targets), utilities = batch

        x = {k: v.to("cuda") for k, v in x.items()}

        (predicted_avg, predicted_std) = self.forward(**x)

        utilities = utilities.to("cuda")
        predicted_avg = predicted_avg.flatten()

        var = predicted_std.flatten()
        loss = self.criterion(predicted_avg, utilities, var=var)

        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, (sources, targets), utilities = batch

        x = {k: v.to("cuda") for k, v in x.items()}
        (predicted_avg, predicted_std) = self.forward(**x)

        utilities = utilities.to("cuda")
        predicted_avg = predicted_avg.flatten()

        var = predicted_std.flatten()
        loss = self.criterion(predicted_avg, utilities, var=var)

        self.log("val_loss", loss)

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

        labels = labels * ~ is_padding + self.padding_id * is_padding

        avg, std = self.forward(input_ids, attention_mask, labels, decoder_input_ids)
        return avg, std

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.linear_layers.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
