### Script for testing a model


import argparse

from datasets import load_metric
from tqdm import tqdm

from custom_datasets.BayesRiskDatasetLoader import BayesRiskDatasetLoader
import numpy as np

from metrics.CometMetric import CometMetric


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test MBR based on pre calculated scores')
    parser.add_argument('--n-hypotheses', type=int, default=100, help='Number of hypothesis to use')
    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')
    parser.add_argument('--utility', type=str, default="unigram-f1")
    parser.add_argument('--n-references', type=int, default=1000, help='Number of references for each hypothesis')

    split = 'validation_predictive'

    args = parser.parse_args()

    dataset_loader = BayesRiskDatasetLoader(split, n_hypotheses=args.n_hypotheses, n_references=args.n_references, utility=args.utility,
                                            sampling_method='ancestral')

    dataset = dataset_loader.load(type="pandas")

    sacreblue_metric = load_metric('sacrebleu')

    comet_metric = CometMetric(model_name="wmt20-comet-da")

    c = 0
    for row in tqdm(dataset.data.iterrows(), total=2500):
        c += 1
        row = row[1]  # Zeroth contains

        target = row["target"]
        source = row["source"]

        utilities = row["utilities"]

        utilities_count = row["utilities_count"]

        best_h = ''
        best_score = - np.infty
        for i, (h, utils_h) in enumerate(zip(row["hypotheses"], utilities)):

            mbr_score = np.sum(utils_h * utilities_count)

            if mbr_score > best_score:
                best_h = h
                best_score = mbr_score

        sacreblue_metric.add_batch(predictions=[best_h], references=[[target]])

        comet_metric.add(source, best_h, target)

    bleu = sacreblue_metric.compute()


    scores = comet_metric.compute()


    test_results = {
        "sacrebleu": bleu,
        "comet": scores
    }




if __name__ == '__main__':
    main()
