### This script is used to generate datasets for predictive modelling

import argparse
from collections import Counter

from datasets import tqdm, Dataset
from torch.utils.data import DataLoader

from utils.config_utils import parse_config
import pandas as pd
import torch

# What to do:
from utils.dataset_utils import get_collate_fn
from utils.translation_model_utils import translate, batch_sample


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str, default='./configs/example.yml',
                        help='config to load model from')
    parser.add_argument('--n-samples', type=int, default=100, help='number of references for each source')
    parser.add_argument('--split', type=str, default="train_predictive",
                        help="Which split to pick (train_predictive or validation_predictive")
    parser.add_argument('--result-dir', type=str, default="./data/",
                        help="Where to save the resulting dataset")
    parser.add_argument('--save-every', type=int, default=100,
                        help="save the intermediate results every n steps (this to make sure that we can continue if we crash for some reason")

    args = parser.parse_args()

    save_every = args.save_every

    n_samples_needed = args.n_samples

    parsed_config = parse_config(args.config, pretrained=True)

    model = parsed_config["model"]
    tokenizer = parsed_config["tokenizer"]
    model = model.to("cuda")
    model.eval()

    dataset = parsed_config["dataset"][args.split].map(
        parsed_config["preprocess_function"], batched=True)
    collate_fn = get_collate_fn(model, tokenizer, parsed_config["config"]["dataset"]["source"],
                                parsed_config["config"]["dataset"]["target"])

    model_name = parsed_config["config"]["name"]
    save_file = "{}{}_{}_{}.csv".format(args.result_dir, args.split, model_name, n_samples_needed)

    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1, shuffle=False)

    column_names = ["source", "target", ] + ["samples"]
    results = pd.DataFrame(columns=column_names)

    with torch.no_grad():

        for i, (x, (sources, targets)) in tqdm(enumerate(dataloader), total=len(dataloader), ):

            sample = batch_sample(model, tokenizer, sources, n_samples=n_samples_needed, batch_size=250, )

            # Add each sample
            for j, (source, target) in enumerate(zip(sources, targets)):
                sample_for_source = sample[j * n_samples_needed: (j + 1) * n_samples_needed]
                counter_samples = dict(Counter(sample_for_source))

                results = results.append({"source": source, "target": target, "samples": counter_samples},
                                         ignore_index=True)

            # Save results every x steps for if we crash

            if (i + 1) % save_every == 0:
                print("saving file at step: {} to {}".format(i, save_file))

                results.to_csv(save_file, index=False, sep="\t")
    print("Saving...")
    results.to_csv(save_file, index=False, sep="\t")
    print("Done!")


if __name__ == '__main__':
    main()
