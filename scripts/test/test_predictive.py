### Script for testing a model


import argparse
from pathlib import Path
from datasets import load_metric
from tqdm import tqdm

from custom_datasets.BayesRiskDatasetLoader import BayesRiskDatasetLoader
from metrics.CometMetric import CometMetric
from metrics.NGramF1Metric import NGramF1Metric
from models.MBR_model.GaussianMBRModel import GaussianMBRModel
from models.MBR_model.GaussianMixtureMBRModel import GaussianMixtureMBRModel
from models.MBR_model.MSEMBRModel import MSEMBRModel

from models.MBR_model.StudentTMixtureMBRModel import StudentTMixtureMBRModel
from models.pl_predictive.GaussianMixturePredictiveModel import GaussianMixturePredictiveModel
from models.pl_predictive.GaussianPredictiveModel import GaussianPredictiveModel
from models.pl_predictive.MSEPredictiveModel import MSEPredictiveModel
from models.pl_predictive.PLPredictiveModelFactory import PLPredictiveModelFactory
from models.pl_predictive.StudentTMixturePredictiveModel import StudentTMixturePredictiveModel
from utils.dataset_utils import save_dict_to_json


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test MBR based on pre calculated scores')
    parser.add_argument('--n-hypotheses', type=int, default=100, help='Number of hypothesis to use')
    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--n-references', type=int, default=1000, help='Number of references for each hypothesis')
    parser.add_argument('--split', type=str, default="validation_predictive")
    parser.add_argument('--model-name', type=str, default='MSE')
    parser.add_argument('--utility', type=str, default="unigram-f1")
    parser.add_argument('--base-dir', type=str, default='C:/Users/gerso/FBR/predictive/tatoeba-de-en/models/')

    args = parser.parse_args()
    path = args.base_dir + args.model_name + '/{}/'.format(args.utility)
    result_save_path = './results/{}/{}'.format(args.model_name, args.utility)

    Path(result_save_path).mkdir(parents=True, exist_ok=True)

    result_save_name = '{}/results.json'.format(result_save_path)

    split = args.split
    dataset_loader = BayesRiskDatasetLoader(split, n_hypotheses=args.n_hypotheses, n_references=args.n_references, utility=args.utility,
                                            sampling_method='ancestral')

    dataset = dataset_loader.load(type="pandas")

    sacreblue_metric = load_metric('sacrebleu')
    comet_metric = CometMetric(model_name="wmt20-comet-da")

    unigram_f1_metric = NGramF1Metric(1)

    pl_model, factory = PLPredictiveModelFactory.load(path)

    pl_model.eval()

    if type(pl_model) == StudentTMixturePredictiveModel:
        model = StudentTMixtureMBRModel(pl_model)
    elif type(pl_model) == GaussianMixturePredictiveModel:
        model = GaussianMixtureMBRModel(pl_model)
    elif type(pl_model) == GaussianPredictiveModel:
        model = GaussianMBRModel(pl_model)
    elif type(pl_model) == MSEPredictiveModel:
        model = MSEMBRModel(pl_model)
    else:
        raise ValueError("model not found")

    c = 0
    for row in tqdm(dataset.data.iterrows(), total=2500):
        c += 1
        row = row[1]  # Zeroth contains index

        source = row["source"]
        target = row["target"]

        hypotheses = list(row["hypotheses"])

        best_h = model.get_best(source, hypotheses)

        sacreblue_metric.add_batch(predictions=[best_h], references=[[target]])
        comet_metric.add(source, best_h, target)
        unigram_f1_metric.add(source, best_h, target)

    bleu = sacreblue_metric.compute()
    #comet_score = comet_metric.compute()
    comet_score = 0
    unigram_score = unigram_f1_metric.compute()
    test_results = {
        "sacrebleu": bleu,
        "comet": comet_score,
        "unigram_f1": unigram_score
    }

    print(test_results)

    save_dict_to_json(test_results, result_save_name)


if __name__ == '__main__':
    main()
