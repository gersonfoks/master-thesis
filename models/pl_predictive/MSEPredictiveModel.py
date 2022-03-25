from torch import nn
from torch.nn import MSELoss

from custom_loss.GaussianMixtureLoss import GaussianMixtureLoss
from models.pl_predictive.PLBasePredictiveModel import PLBasePredictiveModel

from transformers import DataCollatorForSeq2Seq


class MSEPredictiveModel(PLBasePredictiveModel):

    def __init__(self, nmt_model, tokenizer, head, feature_names, initialize_optimizer, feature_map, padding_id=-100,

                 device="cuda", n_mixtures=2):
        super().__init__(nmt_model, tokenizer, head, initialize_optimizer, padding_id=padding_id,
                         device=device, )

        self.n_mixtures = n_mixtures
        self.criterion = MSELoss()

        self.mode = "text"

        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                                    padding=True, return_tensors="pt")
        self.feature_names = feature_names

        self.head = self.head.to(device)

        self.feature_map = feature_map

    def forward(self, input_ids, attention_mask, labels, decoder_input_ids):
        features = self.get_features(input_ids, attention_mask, labels, decoder_input_ids)

        return self.forward_features(features)

    def forward_features(self, features):
        out = self.head.forward(features)

        return out

    def get_predicted_risk(self, input_ids, attention_mask, labels, decoder_input_ids):
        raise NotImplementedError()

    def batch_to_out(self, batch):
        if self.mode == "text":
            x, (sources, targets), utilities = batch

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


    def get_features(self, input_ids, attention_mask, labels, decoder_input_ids):
        nmt_out = self.nmt_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                         decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                                         output_attentions=True)
        nmt_in = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'decoder_input_ids': decoder_input_ids
        }
        return self.feature_map(nmt_in, nmt_out)
