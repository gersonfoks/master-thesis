### This script is used to generate datasets for predictive modelling

import argparse
from collections import Counter
from datetime import datetime

import yaml
from datasets import tqdm, Dataset
from torch.utils.data import DataLoader

from custom_datasets.SampleDataset import SampleDataset

import torch

# What to do:
from utils.PathManager import get_path_manager
from utils.dataset_utils import get_collate_fn, get_dataset
from utils.parsing.predictive import load_nmt_model
from utils.train_utils import preprocess_tokenize
from utils.translation_model_utils import batch_sample


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Generate samples ')
    parser.add_argument('--config', type=str, default='./configs/NMT/helsinki-tatoeba-de-en.yml',
                        help='config to load model from')
    parser.add_argument('--n-samples', type=int, default=10, help='number of references for each source')
    parser.add_argument('--split', type=str, default="train_predictive",
                        help="Which split to generate samples for (train_predictive, validation_predictive or test")

    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    parser.add_argument('--base-dir', type=str, default='NMT/tatoeba-de-en/data/')

    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    args = parser.parse_args()

    n_samples_needed = args.n_samples

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    nmt_model, tokenizer = load_nmt_model(config, pretrained=True)

    model = nmt_model.to("cuda")
    model.eval()

    # load the dataset
    dataset = get_dataset("tatoeba", source="de",
                          target="en")

    dataset = dataset[args.split]

    path_manager = get_path_manager()

    if args.develop:
        dataset = Dataset.from_dict(dataset[:100])
        save_file = "{}{}_{}_{}_develop.csv".format(args.base_dir, args.split, args.sampling_method, n_samples_needed, )
    else:
        save_file = "{}{}_{}_{}.csv".format(args.base_dir, args.split, args.sampling_method, n_samples_needed)

    save_file = path_manager.get_abs_path(save_file)
    print(save_file)

    preprocess_function = lambda x: preprocess_tokenize(x, tokenizer)
    dataset = dataset.map(
        preprocess_function, batched=True)

    collate_fn = get_collate_fn(model, tokenizer, config["dataset"]["source"],
                                config["dataset"]["target"])

    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1, shuffle=False)

    resulting_dataset = SampleDataset(None, args.split)


    with torch.no_grad():

        for i, (x, (sources, targets)) in tqdm(enumerate(dataloader), total=len(dataloader), ):

            sample = batch_sample(model, tokenizer, sources, n_samples=n_samples_needed, batch_size=250, )
            
            # Add each sample
            for j, (source, target) in enumerate(zip(sources, targets)):
                sample_for_source = sample[j * n_samples_needed: (j + 1) * n_samples_needed]
                counter_samples = dict(Counter(sample_for_source))

                resulting_dataset.add_samples(source, target, counter_samples)



    print("Saving to {}".format(save_file))



    resulting_dataset.save(save_file)

    print("Done!")


if __name__ == '__main__':
    main()
