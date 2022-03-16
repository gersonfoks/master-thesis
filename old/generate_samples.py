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
from utils.translation_model_utils import batch_sample


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str, default='./trained_models/NMT/helsinki-tatoeba-de-en.yml',
                        help='config to load model from')
    parser.add_argument('--n-samples', type=int, default=10, help='number of references for each source')
    parser.add_argument('--split', type=str, default="train_predictive",
                        help="Which split to generate samples for (train_predictive, validation_predictive or test")

    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    parser.add_argument('--base-dir', type=str, default='./trained_models/NMT/tatoeba-de-en/data/')

    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    args = parser.parse_args()

    n_samples_needed = args.n_samples

    parsed_config = parse_config(args.config, pretrained=True)

    model = parsed_config["model"]
    tokenizer = parsed_config["tokenizer"]
    model = model.to("cuda")
    model.eval()

    dataset = parsed_config["dataset"][args.split]

    if args.develop:
        dataset = Dataset.from_dict(dataset[:100])
        save_file = "{}{}_{}_{}_develop.csv".format(args.base_dir, args.split, args.sampling_method, n_samples_needed,)
    else:
        save_file = "{}{}_{}_{}.csv".format(args.base_dir, args.split, args.sampling_method, n_samples_needed)

    dataset = dataset.map(
        parsed_config["preprocess_function"], batched=True)

    collate_fn = get_collate_fn(model, tokenizer, parsed_config["config"]["dataset"]["source"],
                                parsed_config["config"]["dataset"]["target"])

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

    print("Saving to {}".format(save_file))
    results.to_csv(save_file, index=False, sep="\t")
    print("Done!")


if __name__ == '__main__':
    main()
