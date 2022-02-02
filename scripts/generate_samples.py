### This script is used to generate datasets for predictive modelling

import argparse
from collections import Counter

from datasets import tqdm
from torch.utils.data import DataLoader

from utils.config_utils import parse_config
import pandas as pd
import torch

# What to do:
from utils.dataset_utils import get_collate_fn
from utils.translation_model_utils import translate


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str, default='./configs/example.yml',
                        help='config to load model from')
    parser.add_argument('--n-hypothesis', type=int, default=10, help='number of hypothesis for each source')
    parser.add_argument('--n-references', type=int, default=100, help='number of references to check the hypothesis against')
    parser.add_argument('-n-reference-sets', type=int, default=10, help='number of sets for reference')

    args = parser.parse_args()

    n_samples_needed = args.n_hypothesis + args.n_references * args.n_reference_sets

    parsed_config = parse_config(args.config, pretrained=True)

    model = parsed_config["model"]
    tokenizer = parsed_config["tokenizer"]
    model = model.to("cuda")
    model.eval()

    dataset = parsed_config["dataset"]["train_predictive"].map(parsed_config["preprocess_function"], batched=True)
    collate_fn = get_collate_fn(model, tokenizer, parsed_config["config"]["dataset"]["source"],
                                parsed_config["config"]["dataset"]["target"])

    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=8)

    column_names = ["hypothesis", ] + ["references_{}".format(i) for i in range(args.n_reference_sets)]
    results = pd.DataFrame(columns=column_names)

    with torch.no_grad():
        for x, (sources, targets) in tqdm(dataloader,):


            for source in sources:
                copied_sources = [source] * 1000

                # Generate samples
                translations = translate(model, tokenizer, copied_sources, batch_size=8, )
                print(Counter(translations))


if __name__ == '__main__':
    main()
