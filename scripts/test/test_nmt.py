### Script for testing a model


import argparse

from datasets import load_metric

from tqdm import tqdm

from custom_datasets.BayesRiskDatasetLoader import BayesRiskDatasetLoader

from utils.parsing.predictive import load_nmt_model
from utils.translation_model_utils import translate


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test MBR based on pre calculated scores')

    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    split = 'test'
    args = parser.parse_args()

    dataset_loader = BayesRiskDatasetLoader(split, n_hypotheses=100, n_references=1000,
                                            sampling_method='ancestral')

    dataset = dataset_loader.load(type="pandas")

    sacreblue_metric = load_metric('sacrebleu')

    config = {
        "model":
            {"name": 'Helsinki-NLP/opus-mt-de-en',
             "checkpoint": 'NMT/tatoeba-de-en/model',
             "type": 'MarianMT'}
    }

    nmt_model, tokenizer = load_nmt_model(config, pretrained=True)

    nmt_model = nmt_model.eval().to("cuda")

    c = 0
    for row in tqdm(dataset.data.iterrows(), total=5000):
        c += 1
        row = row[1]  # Zeroth contains

        source = row["source"]
        target = row["target"]

        hypothesis = translate(nmt_model, tokenizer, [source], method=args.sampling_method)[0]

        sacreblue_metric.add_batch(predictions=[hypothesis], references=[[target]])

    bleu = sacreblue_metric.compute()

    test_results = {
        "sacrebleu": bleu
    }

    print(test_results)


if __name__ == '__main__':
    main()
