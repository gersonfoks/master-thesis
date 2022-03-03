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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test an NMT model')
    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    parser.add_argument('--base-dir', type=str, default='./trained_models/NMT/tatoeba-de-en/data/')

    parser.add_argument('--n-hypothesis', type=int, default=10, help='Number of hypothesis to use')
    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--n-references', type=int, default=100, help='Number of references for each hypothesis')

    parser.add_argument('--split', type=str, default="train_predictive",
                        help="Which split to generate samples for (train_predictive, validation_predictive or test")


    args = parser.parse_args()



    ref_dataset_file = "{}{}_{}_{}".format(args.base_dir, args.split, args.sampling_method,
                                                       args.n_references, )
    hyp_dataset_file = "{}{}_{}_{}".format(args.base_dir, args.split, args.sampling_method,
                                                       args.n_hypothesis, )
    save_file = "{}{}_{}_scores_{}_{}".format(args.base_dir, args.split, args.sampling_method,
                                                          args.n_hypothesis, args.n_references, )

    if args.develop:
        ref_dataset_file += "_develop"
        hyp_dataset_file += "_develop"
        save_file += "_develop"
    ref_dataset_file += ".csv"
    hyp_dataset_file += ".csv"
    save_file += ".csv"

    ref_data = pd.read_csv(ref_dataset_file, sep="\t")
    hyp_data = pd.read_csv(hyp_dataset_file, sep="\t")

    hyp_data["samples"] = hyp_data["samples"].map(lambda x: ast.literal_eval(x))
    ref_data["samples"] = ref_data["samples"].map(lambda x: ast.literal_eval(x))


    results = {
        "sources": [],
        "hypothesis": [],
        "utilities": [],
        "count": []
    }

    # Load the model
    model_path = download_model("wmt21-cometinho-da")
    model = load_from_checkpoint(model_path)

    model.to("cuda")
    #model.eval()

    wrapped_model = CometWrapper(model)
    pl.seed_everything(12)
    model.eval()


    with torch.no_grad():
        for source, references, hypothesis in tzip(ref_data["source"], ref_data["samples"], hyp_data["samples"]):

            reference_sets = [*references.keys()]
            hyp_list = [*hypothesis.keys()]

            scores = wrapped_model.fast_predict(source, hyp_list, reference_sets)
            utilities_list = []
            count_list = []
            for (h, count), score in zip(hypothesis.items(), scores):

                utilities = {}
                count_total = 0
                for (ref, c), s in zip(references.items(), score):
                    s = s.cpu().item()
                    utilities[s] = c
                    count_total += c

                count_list.append(count)



                utilities_list.append(utilities)

            results["sources"].append(source)
            results["hypothesis"].append(hyp_list)
            results["utilities"].append(utilities_list)
            results["count"].append(count_list)



    df =pd.DataFrame.from_dict(results)

    #
    print("Saving...")
    df.to_csv(save_file, index=False, sep="\t")
    print("Done!")


if __name__ == '__main__':
    main()
