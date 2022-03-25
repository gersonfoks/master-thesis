### Script for testing a model


import argparse

from datasets import load_metric
from tqdm import tqdm

from custom_datasets.BayesRiskDatasetLoader import BayesRiskDatasetLoader


from models.MBR_model.GaussianMixtureMBRModel import GaussianMixtureMBRModel
from models.MBR_model.StudentTMixtureMBRModel import StudentTMixtureMBRModel
from models.pl_predictive.PLPredictiveModelFactory import PLPredictiveModelFactory


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test MBR based on pre calculated scores')
    parser.add_argument('--n-hypotheses', type=int, default=100, help='Number of hypothesis to use')
    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--n-references', type=int, default=1000, help='Number of references for each hypothesis')

    split = 'validation_predictive'
    path = "C:/Users/gerso/FBR/predictive/tatoeba-de-en/models/student-t-3/20/"
    args = parser.parse_args()

    dataset_loader = BayesRiskDatasetLoader(split, n_hypotheses=args.n_hypotheses, n_references=args.n_references,
                                            sampling_method='ancestral')

    dataset = dataset_loader.load(type="pandas")

    sacreblue_metric = load_metric('sacrebleu')

    pl_model, factory = PLPredictiveModelFactory.load(path)

    pl_model.eval()
    model = StudentTMixtureMBRModel(pl_model)

    c = 0
    for row in tqdm(dataset.data.iterrows(), total=2500):
        c += 1
        row = row[1]  # Zeroth contains

        source = row["source"]
        target = row["target"]

        hypotheses = list(row["hypotheses"])

        best_h = model.get_best(source, hypotheses)


        sacreblue_metric.add_batch(predictions=[best_h], references=[[target]])


    bleu = sacreblue_metric.compute()

    test_results = {
        "sacrebleu": bleu
    }

    print(test_results)


if __name__ == '__main__':
    main()
