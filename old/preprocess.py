# File to train a model from scratch
import pickle
import argparse

import numpy as np

from datasets import tqdm
from torch.utils.data import DataLoader
import torch
import pandas as pd

from utils.config_utils import parse_predictive_config
from utils.dataset_utils import get_preprocess_collate_fn


def get_preprocess_function(tokenizer):
    def preprocess_function(examples, tokenizer, ):
        source = examples["source"]
        targets = examples["hypothesis"]
        model_inputs = tokenizer(source, truncation=True, )
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, truncation=True, )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return lambda examples: preprocess_function(examples, tokenizer)


def preprocess_dataset(dataset, model, save_location):
    columns = ["source", "hypothesis", "utilities", "avg_pool_encoder_hidden_state",
               "max_pool_encoder_hidden_state",
               "avg_pool_decoder_hidden_state",
               "max_pool_decoder_hidden_state", "count"]
    #new_dataset = pd.DataFrame()

    data = {
        col: [] for col in columns
    }

    with torch.no_grad():
        for x, (sources, targets), utilities, counts in tqdm(dataset):

            x = {k: v.to("cuda") for k, v in x.items()}

            features = model.get_features(**x)

            i = 0

            for source, target, utility, count in zip(sources, targets, utilities, counts):


                data["source"].append(source)
                data["hypothesis"].append(source)
                data["utilities"].append(utility)
                data["avg_pool_encoder_hidden_state"].append(features["avg_pool_encoder_hidden_state"][
                        i].cpu().numpy().tolist())
                data["max_pool_encoder_hidden_state"].append(features["max_pool_encoder_hidden_state"][
                        i].cpu().numpy().tolist())
                data["avg_pool_decoder_hidden_state"].append(features["avg_pool_decoder_hidden_state"][
                        i].cpu().numpy().tolist())
                data["max_pool_decoder_hidden_state"].append(features["max_pool_decoder_hidden_state"][
                        i].cpu().numpy().tolist())
                data["count"].append(count)

                # new_data_row = {
                #     "source": source,
                #     "hypothesis": target,
                #     "utilities": utility,
                #     "avg_pool_encoder_hidden_state":features["avg_pool_encoder_hidden_state"][
                #         i].cpu().numpy().tolist(),
                #     "max_pool_encoder_hidden_state": features["max_pool_encoder_hidden_state"][
                #         i].cpu().numpy().tolist(),
                #     "avg_pool_decoder_hidden_state": features["avg_pool_decoder_hidden_state"][
                #         i].cpu().numpy().tolist(),
                #     "max_pool_decoder_hidden_state": features["max_pool_decoder_hidden_state"][
                #         i].cpu().numpy().tolist(),
                #     "count": count
                # }
                i += 1

                # temp_df = pd.DataFrame(
                #     columns=["source", "hypothesis", "utilities", "avg_pool_encoder_hidden_state",
                #              "max_pool_encoder_hidden_state",
                #              "avg_pool_decoder_hidden_state",
                #              "max_pool_decoder_hidden_state", "count"])



                #new_dataset= pd.concat([new_dataset, temp_df], ignore_index=True, copy=False)
    new_dataset = pd.DataFrame.from_dict(data)  # temp_df.append(new_data_row, ignore_index=True, )

    new_dataset.to_csv(save_location, index=False, sep="\t")


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str, default='./trained_models/predictive/tatoeba-de-en-predictive.yml',
                        help='config to load model from')
    parser.add_argument('--save-loc', type=str, default='./trained_models/predictive/tatoeba-de-en/data/',
                        help='config to load model from')

    args = parser.parse_args()

    train_save = args.save_loc + 'preprocessed_train.csv'
    validation_save = args.save_loc + 'preprocessed_validation.csv'

    parsed_config = parse_predictive_config(args.config, pretrained=False, preprocessing=True )
    nmt_model = parsed_config["nmt_model"]
    pl_model = parsed_config["pl_model"]
    tokenizer = parsed_config["tokenizer"]
    train_dataset = parsed_config["train_dataset"]

    val_dataset = parsed_config["validation_dataset"]

    preprocess_function = get_preprocess_function(tokenizer)
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)

    collate_fn = get_preprocess_collate_fn(nmt_model, tokenizer, )

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=32, shuffle=False, )
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=32, shuffle=False, )

    # Create a new dataset of the form:
    preprocess_dataset(train_dataloader, pl_model, train_save)

    preprocess_dataset(val_dataloader, pl_model, validation_save)


if __name__ == '__main__':
    main()
