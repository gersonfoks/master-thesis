### Script for testing a model


import argparse
import ast
import torch
from collections import Counter

import pandas as pd
from comet import download_model, load_from_checkpoint
import numpy as np
from datasets import tqdm

from utils.translation_model_utils import batch
from models.wrappers.CometWrapper import CometWrapper


def counter_to_relative(counter):
    samples_with_freq = [(k, v) for k, v in counter.items()]
    samples = [k for (k, v) in samples_with_freq]
    freqs = np.array([v for (k, v) in samples_with_freq])
    probs = freqs / freqs.sum()

    return samples, freqs, probs


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test an NMT model')
    parser.add_argument('--dataset', type=str, default='./data/train_predictive_helsinki-tatoeba-de-en_1000.csv',
                        help='dataset to load')
    parser.add_argument('--n-references', type=int, default=100,
                        help='dataset to load')
    parser.add_argument('--n-reference-sets', type=str, default=10,
                        help='dataset to load')
    parser.add_argument('--n-hypothesis', type=str, default=5,
                        help='dataset to load')

    args = parser.parse_args()

    save_file = args.dataset[:-4] + '_bayes_risk.csv'

    # Load the data
    data = pd.read_csv(args.dataset, sep="\t")
    print(save_file)
    # Dataframe to save the results in
    columns = ["source", "hypothesis", "utilities"]
    df = pd.DataFrame(columns=columns)

    # Load the model
    model_path = download_model("wmt21-cometinho-da")
    model = load_from_checkpoint(model_path)

    model.to("cuda")
    model.eval()
    wrapped_model = CometWrapper(model)
    with torch.no_grad():
        for i, row in tqdm(data.iterrows(), total=data.shape[0]):
            source = row["source"]
            samples_counter = ast.literal_eval(row["samples"])

            # First pick random n hypothesis (without replacement)
            samples, freqs, probs = counter_to_relative(samples_counter)

            size = args.n_hypothesis

            # We skip the once that we don't have enough
            if len(samples) < args.n_hypothesis:
                size = len(samples)
            hypothesis = np.random.choice(samples, size=size, p=probs, replace=False, )

            # Next we randomly divide the set into a number of reference sets
            new_samples = []
            for k, v in zip(samples, freqs):
                new_samples += [k] * v

            # Next we shuffle the dataset and break it up into n_reference sets
            np.random.shuffle(new_samples)

            reference_sets = []
            for j in range(args.n_reference_sets):
                reference_sets.append(Counter(new_samples[j * args.n_references: (j + 1) * args.n_references]))

            ### Calculate the bayes Risk for every hypothesis compared with the every reference set

            # Calculate the utility of each hypothesis, compared to each reference

            utilities = {}
            for h in hypothesis:
                data = [

                ]
                for reference in samples:
                    data.append(
                        {"src": source, "mt": h, "ref": reference}
                    )

                # Now we will loop over the data and start predicting the score

                for batched_data in batch(data, n=32):

                    # Get the scores
                    scores = wrapped_model.predict(batched_data, )["score"].flatten().cpu().numpy()

                    # Put the scores in the dictionary
                    for data_point, score in zip(batched_data, scores):
                        utilities[(data_point["mt"], data_point["ref"])] = score

            # Next we will compute bayes risk

            hypothesis_scores = {h: [] for h in hypothesis}
            for reference_set_number, reference_set in enumerate(reference_sets):
                # Make frequencies relative:
                samples, freqs, probs = counter_to_relative(reference_set)

                for h in hypothesis:
                    weighted_score = 0
                    for s, p in zip(samples, probs):
                        # Get weighted score
                        weighted_score += utilities[(h, s)] * p
                    hypothesis_scores[h].append(weighted_score)

            # statistics = {h: (np.mean(risks), np.std(risks)) for }

            for n_h, (h, risks) in enumerate(hypothesis_scores.items()):
                new_data_row = {"source": source}
                new_data_row["hypothesis".format(n_h)] = h
                new_data_row["utilities".format(n_h)] = risks
                df = df.append(new_data_row, ignore_index=True)

    print("Saving...")
    df.to_csv(save_file, index=False, sep="\t")
    print("Done!")


if __name__ == '__main__':
    main()
