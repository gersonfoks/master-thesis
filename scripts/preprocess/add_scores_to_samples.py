import argparse

import torch

from tqdm import tqdm

from custom_datasets.BayesRiskDataset.BayesRiskDatasetCreator import BayesRiskDatasetCreator
from custom_datasets.SampleDataset.SampleDatasetLoader import SampleDatasetLoader

from utilities.utilities import NGramF


def load_utility(utility):
    if utility == "unigram-f1":
        return NGramF(1, tokenize=True)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Give the COMET scores for hypothesis given a reference set')
    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    parser.add_argument('--base-dir', type=str, default='NMT/tatoeba-de-en/data/')

    parser.add_argument('--save-dir', type=str, default='predictive/tatoeba-de-en/data/raw/')

    parser.add_argument('--utility', type=str, default="unigram-f1")

    parser.add_argument('--n-hypotheses', type=int, default=10, help='Number of hypothesis to use')
    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--n-references', type=int, default=100, help='Number of references for each hypothesis')

    parser.add_argument('--split', type=str, default="train_predictive",
                        help="Which split to generate samples for (train_predictive, validation_predictive or test")

    args = parser.parse_args()

    ref_dataset_loader = SampleDatasetLoader(args.split, args.n_references, args.sampling_method, develop=args.develop,
                                             base=args.base_dir, )
    hyp_dataset_loader = SampleDatasetLoader(args.split, args.n_hypotheses, args.sampling_method, develop=args.develop,
                                             base=args.base_dir, )

    ref_dataset = ref_dataset_loader.load()
    hyp_dataset = hyp_dataset_loader.load()

    # We get an empty dataset
    dataset_creator = BayesRiskDatasetCreator(args.split, args.n_hypotheses, args.n_references,
                                                      args.sampling_method, args.utility, develop=args.develop,
                                                      base=args.save_dir,)

    utility = load_utility(args.utility)

    with torch.no_grad():
        pbar = tqdm(total=len(hyp_dataset))
        i = 0
        for (_, hyp_data), (_, ref_data) in zip(hyp_dataset, ref_dataset):
            i += 1
            source = hyp_data["source"]

            references = ref_data["samples"]

            hyp_list = hyp_data["samples"]

            scores = utility.call_batched(source, hyp_list, references)

            dataset_creator.add_row(source, hyp_data["target"], hyp_data["samples"], scores, ref_data["count"],
                                      hyp_data["count"],
                                      )

            pbar.update(1)

        pbar.close()

    dataset_creator.save()


if __name__ == '__main__':
    main()
