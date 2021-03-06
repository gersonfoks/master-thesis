### Script for testing a model


import argparse

from datasets import load_metric

from tqdm import tqdm

from custom_datasets.BayesRiskDatasetLoader import BayesRiskDatasetLoader
from metrics.CometMetric import CometMetric
from metrics.NGramF1Metric import NGramF1Metric

from utils.parsing.predictive import load_nmt_model
from utils.translation_model_utils import translate


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test NMT model')

    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')
    parser.add_argument('--utility', type=str, default="unigram-f1")
    split = 'validation_predictive'
    args = parser.parse_args()

    dataset_loader = BayesRiskDatasetLoader(split, n_hypotheses=100, n_references=1000, utility=args.utility,
                                            sampling_method='ancestral')

    dataset = dataset_loader.load(type="pandas")

    sacreblue_metric = load_metric('sacrebleu')
    comet_metric = CometMetric(model_name="wmt20-comet-da")
    unigram_f1_metric = NGramF1Metric(1)
    config = {
        "model":
            {"name": 'Helsinki-NLP/opus-mt-de-en',
             "checkpoint": 'NMT/tatoeba-de-en/model',
             "type": 'MarianMT'}
    }

    nmt_model, tokenizer = load_nmt_model(config, pretrained=True)

    nmt_model = nmt_model.eval().to("cuda")



    c = 0
    for row in tqdm(dataset.data.iterrows(), total=2500):
        c += 1
        row = row[1]  # Zeroth contains

        source = row["source"]
        target = row["target"]

        hypothesis = translate(nmt_model, tokenizer, [source], method=args.sampling_method)[0]

        sacreblue_metric.add_batch(predictions=[hypothesis], references=[[target]])
        comet_metric.add(source, hypothesis, target)
        unigram_f1_metric.add(source, hypothesis, target)

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
