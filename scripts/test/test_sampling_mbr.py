### Script for testing a model


import argparse
import math

from datasets import load_metric
from tqdm import tqdm

from custom_datasets.BayesRiskDatasetLoader import BayesRiskDatasetLoader
import numpy as np

from metrics.CometMetric import CometMetric
import pandas as pd

from metrics.NGramF1Metric import NGramF1Metric


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test model used as a predictive filter')
    parser.add_argument('--n-hypotheses', type=int, default=100, help='Number of hypothesis to use')
    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--top-p', type=float, default=0.2, help="What percentage of the best scoring hypotheses we should keep to test mbr score")

    parser.add_argument('--n-references', type=int, default=1000, help='Number of references for each hypothesis')

    parser.add_argument('--utility', type=str, default="unigram-f1")

    split = 'validation_predictive'

    args = parser.parse_args()

    dataset_loader = BayesRiskDatasetLoader(split, n_hypotheses=args.n_hypotheses, n_references=args.n_references, utility=args.utility,
                                            sampling_method='ancestral')

    dataset = dataset_loader.load(type="pandas")

    sacreblue_metric = load_metric('sacrebleu')

    comet_metric = CometMetric(model_name="wmt20-comet-da")

    unigram_f1_metric = NGramF1Metric(1)





    c = 0
    for row in tqdm(dataset.data.iterrows(), total=2500):
        c += 1
        row = row[1]  # Zeroth contains

        target = row["target"]
        source = row["source"]


        utilities = np.array(row["utilities"])

        utilities_count = row["utilities_count"]



        top_p = math.ceil(args.top_p * len(utilities))



        hypotheses = row["hypotheses"]

        counts = np.array(row["count"])
        probs = counts/np.sum(counts)
        random_indices = np.random.choice(len(utilities), top_p, p=probs, replace=True)


        top_p_hypotheses = hypotheses[random_indices]
        top_p_utilities = utilities[random_indices]
        best_h = ''
        best_score = - np.infty
        for i, (h, utils_h) in enumerate(zip(top_p_hypotheses, top_p_utilities)):

            mbr_score = np.sum(utils_h * utilities_count)

            if mbr_score > best_score:
                best_h = h
                best_score = mbr_score

        sacreblue_metric.add_batch(predictions=[best_h], references=[[target]])

        comet_metric.add(source, best_h, target)
        unigram_f1_metric.add(source, best_h, target)




    bleu = sacreblue_metric.compute()
    # comet_score = comet_metric.compute()
    comet_score = 0
    unigram_score = unigram_f1_metric.compute()

    test_results = {
        "sacrebleu": bleu,
        "comet": comet_score,
        "unigram_f1": unigram_score
    }

    print(test_results)


if __name__ == '__main__':
    main()
