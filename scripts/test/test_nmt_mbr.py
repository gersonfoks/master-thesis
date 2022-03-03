### Script for testing a model


import argparse
import torch
from datasets import tqdm, load_metric
from torch.utils.data import DataLoader
import pandas as pd
from utils.config_utils import parse_config
from utils.dataset_utils import save_dict_to_json, get_collate_fn
from utils.translation_model_utils import translate
import ast
import numpy as np


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test NMT model with MBR with preselected samples')
    parser.add_argument('--config', type=str, default='./trained_models/NMT/helsinki-tatoeba-de-en.yml',
                        help='config to load model from')
    parser.add_argument('--utilities', type=str,
                        default='./trained_models/NMT/tatoeba-de-en/data/test_ancestral_scores_10_100.csv',
                        help='config to load model from')

    args = parser.parse_args()

    utilities = pd.read_csv(args.utilities, sep="\t")
    utilities["utilities"] = utilities["utilities"].apply(lambda x: ast.literal_eval(x))

    # Parse the config and load all the things we need
    parsed_config = parse_config(args.config, pretrained=True)
    model = parsed_config["model"]
    tokenizer = parsed_config["tokenizer"]
    model = model.to("cuda")
    model.eval()
    # Next we preprocess the dataset
    test_dataset = parsed_config["dataset"]["test"].map(parsed_config["preprocess_function"], batched=True)
    collate_fn = get_collate_fn(model, tokenizer, parsed_config["config"]["dataset"]["source"],
                                parsed_config["config"]["dataset"]["target"])

    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=1)

    # Evaluate the model
    sacreblue_metric = load_metric('sacrebleu')
    # First accumulate the metrics and then calculate the average.
    with torch.no_grad():
        for x, (sources, targets) in tqdm(test_dataloader):
            target = targets[0]
            source = sources[0]
            rows = utilities.loc[utilities['source'] == source]

            highest_score = - np.inf
            translation = None
            for row in rows.itertuples():

                score = sum([s * count for s, count in row.utilities.items()])
                if score > highest_score:
                    highest_score = score
                    translation = row.hypothesis

            targets = [[target]]

            sacreblue_metric.add_batch(predictions=[translation], references=targets)

        # Save the results

        bleu = sacreblue_metric.compute()

        test_results = {
            "sacrebleu": bleu
        }

        print(test_results)


if __name__ == '__main__':
    main()
