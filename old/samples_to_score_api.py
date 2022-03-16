import argparse
import ast
import torch

import pandas as pd
from comet import download_model, load_from_checkpoint
import numpy as np

from tqdm.contrib import tzip

from utils.translation_model_utils import batch
from models.wrappers.CometWrapper import CometWrapper
import pytorch_lightning as pl

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
                        default='./trained_models/NMT/tatoeba-de-en/data/train_predictive_ancestral_100_develop.csv',
                        help='The references to use')
    parser.add_argument('--hypothesis-dataset', type=str,
                        default='./trained_models/NMT/tatoeba-de-en/data/train_predictive_ancestral_100_develop.csv',
                        help='The hypothesis to use')
    parser.add_argument("--save-file", type=str,
                        default='./trained_models/NMT/tatoeba-de-en/data/train_predictive_ancestral_scores_100_1000_new.csv')

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
    pl.seed_everything(12)
    model.eval()

    max_count = 1
    with torch.no_grad():
        c = 0
        samples = []
        for source, references, hypothesis in tzip(ref_data["source"], ref_data["samples"], hyp_data["samples"]):
            c += 1


            for h, count in hypothesis.items():

                for ref in references:
                    samples.append(
                        {"src": source, "mt": h, "ref": ref}
                    )
            #prepared_samples = self.model.prepare

            if max_count <= c:

                c = 0
                scores = model.predict(samples, gpus=1, num_workers=0, batch_size=256, progress_bar=True)
                samples = []






if __name__ == '__main__':
    main()
