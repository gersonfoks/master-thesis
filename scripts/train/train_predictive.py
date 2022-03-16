# File to train a model from scratch

import argparse
import pytorch_lightning as pl
import yaml

from torch.utils.data import DataLoader

from callbacks.predictive_callbacks import MyShuffleCallback
from custom_datasets.PreBayesRiskDataset import PreBayesRiskDatasetLoader
from models.pl_predictive.PLPredictiveModelFactory import PLPredictiveModelFactory
from scripts.Collate import Collator, mean_util, SequenceCollator, util_functions


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str, default='./configs/predictive/tatoeba-de-en-cross-attention-gaussian.yml',
                        help='config to load model from')

    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    pl_factory = PLPredictiveModelFactory(config['model_config'])
    pl_model = pl_factory.create_model()
    pl_model.set_mode('features')
    dataset_config = config["dataset"]

    preprocess_dir = dataset_config["preprocess_dir"] + "{}_{}/".format(dataset_config["n_hypotheses"], dataset_config["n_references"])



    bayes_risk_dataset_loader_train = PreBayesRiskDatasetLoader(preprocess_dir, "train_predictive", pl_model.feature_names,
                                                   develop=args.develop, max_chunk_size=4096 * 16, repeated_indices=dataset_config["repeated_indices"])
    bayes_risk_dataset_loader_val = PreBayesRiskDatasetLoader(preprocess_dir, "validation_predictive",
                                                   pl_model.feature_names,
                                                   develop=args.develop, max_chunk_size=4096 * 24, repeated_indices=dataset_config["repeated_indices"])

    train_dataset = bayes_risk_dataset_loader_train.load()
    val_dataset = bayes_risk_dataset_loader_val.load()

    train_dataset.shuffle()

    util_function = util_functions[config['model_config']["loss_function"]]


    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=SequenceCollator(pl_model.feature_names, util_function),
                                  batch_size=config["batch_size"], shuffle=False, )
    val_dataloader = DataLoader(val_dataset,
                                  collate_fn=SequenceCollator(pl_model.feature_names, util_function),
                                  batch_size=config["batch_size"] * 4, shuffle=False, )

    trainer = pl.Trainer(
        max_epochs=4,
        gpus=1,
        progress_bar_refresh_rate=1,
        val_check_interval=0.5,
        callbacks=[MyShuffleCallback(train_dataset)]
    )

    # create the dataloaders
    print("train_first")

    trainer.fit(pl_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    path = './'
    pl_factory.save(pl_model, path)


if __name__ == '__main__':
    main()
