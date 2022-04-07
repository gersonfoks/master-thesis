### Script for testing a model


import argparse
import math

from datasets import load_metric
from tqdm import tqdm

from custom_datasets.BayesRiskDatasetLoader import BayesRiskDatasetLoader
import numpy as np

from metrics.CometMetric import CometMetric
import pandas as pd

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test model used as a predictive filter')
    parser.add_argument('--n-hypotheses', type=int, default=100, help='Number of hypothesis to use')
    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--top-p', type=float, default=0.1, help="What percentage of the best scoring hypotheses we should keep to test mbr score")

    parser.add_argument('--model-name', type=str, default='gaussian-3-repeated')
    parser.add_argument('--base-dir', type=str, default='C:/Users/gerso/FBR/predictive/tatoeba-de-en/models/')

    parser.add_argument('--n-references', type=int, default=1000, help='Number of references for each hypothesis')


    split = 'validation_predictive'

    args = parser.parse_args()

    dataset_loader = BayesRiskDatasetLoader(split, n_hypotheses=args.n_hypotheses, n_references=args.n_references,
                                            sampling_method='ancestral')

    dataset = dataset_loader.load(type="pandas")

    sacreblue_metric = load_metric('sacrebleu')

    comet_metric = CometMetric(model_name="wmt20-comet-da")


    samples_path = args.base_dir + args.model_name + "/samples.parquet"


    samples = pd.read_parquet(samples_path)

    dataset.data["samples"] = samples["samples"]


    # Calculate the means

    def to_mean(x):
        samples = x["samples"]
        result = []
        for s in samples:
            result.append(np.mean(s))
        return result

    dataset.data["predicted_means"] = dataset.data.apply(to_mean, axis=1)

    c = 0
    for row in tqdm(dataset.data.iterrows(), total=2500):
        c += 1
        row = row[1]  # Zeroth contains

        target = row["target"]
        source = row["source"]

        utilities = np.array(row["utilities"])

        utilities_count = row["utilities_count"]

        means = np.array(row["predicted_means"])

        top_p = math.ceil(args.top_p * len(means))

        sorted_indices = np.argsort(-means)[:top_p]



        top_p_utilities = utilities[sorted_indices]
        top_p_hypotheses = np.array(row["hypotheses"])[sorted_indices]

        best_h = ''
        best_score = - np.infty
        for i, (h, utils_h) in enumerate(zip(top_p_hypotheses, top_p_utilities)):

            mbr_score = np.sum(utils_h * utilities_count)

            if mbr_score > best_score:
                best_h = h
                best_score = mbr_score

        sacreblue_metric.add_batch(predictions=[best_h], references=[[target]])

        comet_metric.add(source, best_h, target)

    bleu = sacreblue_metric.compute()


    comet_score = comet_metric.compute()


    test_results = {
        "sacrebleu": bleu,
        "comet": comet_score
    }

    print(test_results)


if __name__ == '__main__':
    main()
