'''
Temp Helper file
'''
import argparse

import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_datasets.BayesRiskDataset.BayesRiskDatasetLoader import BayesRiskDatasetLoader
from custom_datasets.PreprocessedBayesRiskDataset.PreBayesRiskDatasetCreator import PreBayesDatasetCreator
from models.estimation.PLPredictiveModelFactory import PLPredictiveModelFactory


def collate_fn(batch):
    keys = list(batch[0].keys())
    collated = {key: [] for key in keys}
    for e in batch:
        for key in keys:
            collated[key].append(e[key])

    return collated


def main():
    parser = argparse.ArgumentParser(description='Preprocesses a dataset')

    parser.add_argument('--config', type=str,
                        default='./configs/predictive/unigram_f1/cross-attention-MSE.yml',
                        help='config to load model from')
    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    parser.add_argument('--dataset-dir', type=str, default='predictive/tatoeba-de-en/data/raw/')

    parser.add_argument('--utility', type=str, default="unigram-f1")

    parser.add_argument('--save-dir', type=str, default='predictive/tatoeba-de-en/data/preprocessed/')

    parser.add_argument('--n-hypotheses', type=int, default=10, help='Number of hypothesis to use')
    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--n-references', type=int, default=100, help='Number of references for each hypothesis')
    parser.add_argument('--max-dataset-size', type=int, default=4096 * 8, help='Max number of dataset entries ')

    parser.add_argument('--split', type=str, default="train_predictive",
                        help="Which split to generate samples for (train_predictive, validation_predictive or test")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    save_dir = args.save_dir + '/' + args.utility + '/'

    bayes_risk_dataset_loader = BayesRiskDatasetLoader(args.split, args.n_hypotheses, args.n_references,
                                                       args.sampling_method, args.utility, develop=args.develop, base=dataset_dir)

    bayes_risk_dataset = bayes_risk_dataset_loader.load(type="pandas")

    # We want to transform this thing to a extended dataset. We use pandas for this

    df = bayes_risk_dataset.data

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    pl_factory = PLPredictiveModelFactory(config['model_config'])
    pl_model = pl_factory.create_model()
    pl_model.eval()

    # Next we "explode the data
    df_exploded = df.explode(["hypotheses", "utilities", "count"], ignore_index=True)

    # Lastly we create a huggingface dataset from it and

    dataset = Dataset.from_pandas(df_exploded)

    # Next we can start iterating trough the dataset

    dataset_loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)

    # Create the dataset in which we are going to store the results
    max_dataset_size = args.max_dataset_size

    if args.develop:
        max_dataset_size = 100

    dataset_creator = PreBayesDatasetCreator(save_dir, pl_model.feature_names, n_hypotheses=args.n_hypotheses,
                                             n_references=args.n_references, max_dataset_size=max_dataset_size,
                                             split=args.split, develop=args.develop)

    for data in tqdm(dataset_loader):
        preprocessed = pl_model.preprocess_function(data)

        dataset_creator.add_rows({**data, **preprocessed})

    dataset_creator.finalize()


if __name__ == '__main__':
    main()
