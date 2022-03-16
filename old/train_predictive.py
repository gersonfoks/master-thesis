# File to train a model from scratch

import argparse
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from callbacks.predictive_callbacks import MyShuffleCallback
from custom_datasets.PreprocessedBayesRiskDatasetLoader import PreprocessedBayesRiskDatasetLoader
from models.pl_predictive.PLPredictiveModelFactory import PLPredictiveModelFactory
from scripts.Collate import Collator, mean_util, SequenceCollator


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str, default='./configs/predictive/tatoeba-de-en-cross-attention.yml',
                        help='config to load model from')

    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    pl_factory = PLPredictiveModelFactory(config['model_config'])
    pl_model = pl_factory.create_model()

    dataset_config = config["dataset"]

    bayes_risk_dataset = PreprocessedBayesRiskDatasetLoader(dataset_config, pl_model, develop=args.develop)

    datasets = bayes_risk_dataset.load()

    pl_model.set_mode('features')

    train_dataloader = DataLoader(datasets["train_predictive"],
                                  collate_fn=SequenceCollator(pl_model.feature_names, mean_util),
                                  batch_size=config["batch_size"], shuffle=False, )
    val_dataloader = DataLoader(datasets["validation_predictive"],
                                collate_fn=SequenceCollator(pl_model.feature_names, mean_util),
                                batch_size=config["batch_size"], shuffle=False, )

    trainer = pl.Trainer(
        max_epochs=4,
        gpus=1,
        progress_bar_refresh_rate=1,
        callbacks=[LearningRateMonitor(), MyShuffleCallback(datasets["train_predictive"])]
    )

    # create the dataloaders
    trainer.fit(pl_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, )

    # Save model

    path = './'
    pl_factory.save(pl_model, path)
    # Load model and
    loaded_pl_model, loaded_pl_factory = PLPredictiveModelFactory.load(path)
    #
    print("start validating")
    # We need to make sure we use features instead of text as input
    loaded_pl_model.set_mode('features')
    trainer.validate(loaded_pl_model, val_dataloader)


if __name__ == '__main__':
    main()
