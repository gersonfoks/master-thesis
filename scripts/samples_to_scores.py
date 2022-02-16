import argparse
import ast
import torch

import pandas as pd
from comet import download_model, load_from_checkpoint
import numpy as np

from tqdm.contrib import tzip

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
    parser.add_argument('--ref-dataset', type=str,
                        default='./trained_models/NMT/tatoeba-de-en/data/train_predictive_ancestral_100.csv',
                        help='The references to use')
    parser.add_argument('--hypothesis-dataset', type=str,
                        default='./trained_models/NMT/tatoeba-de-en/data/train_predictive_ancestral_10.csv',
                        help='The hypothesis to use')
    parser.add_argument("--save-file", type=str,
                        default='./trained_models/NMT/tatoeba-de-en/data/train_predictive_ancestral_scores_10_100.csv')

    args = parser.parse_args()

    # Load the data

    ref_data = pd.read_csv(args.ref_dataset, sep="\t")
    hyp_data = pd.read_csv(args.hypothesis_dataset, sep="\t")

    hyp_data["samples"] = hyp_data["samples"].map(lambda x: ast.literal_eval(x))
    ref_data["samples"] = ref_data["samples"].map(lambda x: ast.literal_eval(x))

    # Dataframe to save the results in
    columns = ["source", "hypothesis", "utilities", "count"]
    df = pd.DataFrame(columns=columns)

    # Load the model
    model_path = download_model("wmt21-cometinho-da")
    model = load_from_checkpoint(model_path)

    model.to("cuda")
    model.eval()
    wrapped_model = CometWrapper(model)

    with torch.no_grad():
        for source, references, hypothesis in tzip(ref_data["source"], ref_data["samples"], hyp_data["samples"]):

            reference_sets = [*references.keys()]

            for h, count in hypothesis.items():

                data = [

                ]
                for reference in reference_sets:
                    data.append(
                        {"src": source, "mt": h, "ref": reference}
                    )

                # Now we will loop over the data and start predicting the score

                for batched_data in batch(data, n=32):

                    # Get the scores
                    scores = wrapped_model.predict(batched_data, )["score"].flatten().cpu().numpy()

                    # Unwrap the utilities
                    utilities = {}
                    for data_point, score in zip(batched_data, scores):

                        if score not in utilities.keys():
                            utilities[score] = references[data_point["ref"]]
                        else:
                            utilities[score] += references[data_point["ref"]]

                # Add the new hypothesis with the counted utilities to the final list
                new_data_row = {"source": source}
                new_data_row["hypothesis"] = h
                new_data_row["utilities"] = utilities
                new_data_row["count"] = count
                df = df.append(new_data_row, ignore_index=True)

    print("Saving...")
    df.to_csv(args.save_file, index=False, sep="\t")
    print("Done!")


if __name__ == '__main__':
    main()
