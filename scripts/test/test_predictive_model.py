### Script for testing a model


import argparse
import torch
from datasets import tqdm, load_metric, Dataset
from torch.utils.data import DataLoader
import pandas as pd
from transformers import DataCollatorForSeq2Seq

from utils.config_utils import parse_config, load_model, get_predictive_dataset
from utils.dataset_utils import save_dict_to_json, get_collate_fn
from utils.translation_model_utils import translate
import ast
import numpy as np


def get_preprocess_function(tokenizer):
    def preprocess_function(examples, tokenizer, ):
        targets = examples["hypothesis"]
        source = [examples["source"]] * len(targets)
        model_inputs = tokenizer(source, truncation=True, )
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, truncation=True, )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return lambda examples: preprocess_function(examples, tokenizer)


def get_collate_fn_predictive_test(model, tokenizer, ):
    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer,
                                           padding=True, return_tensors="pt")

    keys = [
        "input_ids",
        "attention_mask",
        "labels"
    ]

    def collate_fn(batch):
        # In this case we go example by example
        batch = batch[0]

        hypothesis = batch["hypothesis"]

        n_samples = len(hypothesis)

        source = batch["source"]
        new_batch = [{} for i in range(n_samples)]

        for i in range(n_samples):

            for k in keys:
                new_batch[i][k] = batch[k][i]

        x_new = data_collator(new_batch)

        # Group the averages and the standard deviations
        utilities = torch.Tensor(batch["utility"])

        return x_new, (source, hypothesis), utilities

    return collate_fn


def get_rank(np_array):
    order = np_array.argsort()

    rank = order.argsort()
    return rank


def main():
    # Training settings
    # parser = argparse.ArgumentParser(description='Test a predictive model')

    # Load the dataset and preprocess
    dataset = pd.read_csv('./data/validation_predictive_scores_5_1000_old.csv',
                          sep="\t")
    dataset["utility"] = dataset["utilities"].apply(lambda x: float(np.mean(ast.literal_eval(x))))

    grouped_dataset = dataset.groupby("source")

    grouped_hypothesis = grouped_dataset["hypothesis"].apply(list).reset_index()
    grouped_utility = grouped_dataset["utility"].apply(list).reset_index()

    # dataset = pd.concat([grouped_hypothesis, grouped_utility], axis=1)
    dataset = grouped_hypothesis.set_index("source").join(grouped_utility.set_index("source"),
                                                          on="source").reset_index()

    model = load_model("./data/develop_model")
    model.eval()
    tokenizer = model.tokenizer

    f = get_preprocess_function(tokenizer)

    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.map(f)

    # Next we create the dataloader.

    collate_fn = get_collate_fn_predictive_test(model.nmt_model, tokenizer, )

    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1, shuffle=False, )

    sum_loss = 0
    accuracy_count = 0
    top_2_acc_count = 0

    with torch.no_grad():
        for x, (source, hypothesis), utilities in tqdm(dataloader):
            x = {k: v.to("cuda") for k, v in x.items()}

            predicted_score = model.forward(**x).flatten()

            utilities = utilities.to("cuda")


            loss = model.criterion(predicted_score, utilities, )
            sum_loss += loss



            # Get the order of everything.
            real_order = utilities.cpu().numpy().argsort().argsort()
            predicted_order = predicted_score.cpu().numpy().argsort().argsort()

            predicted_first = np.argmax(predicted_order)
            real_first = np.argmax(real_order)

            real_second = np.argwhere(real_order == real_order.shape[0]-2)[0][0]

            accuracy_count += predicted_first == real_first
            top_2_acc_count += predicted_first == real_first or predicted_first == real_second

        print(sum_loss / len(dataloader))
        print("acc")
        print(accuracy_count / len(dataloader))
        print(top_2_acc_count / len(dataloader))


if __name__ == '__main__':
    main()
