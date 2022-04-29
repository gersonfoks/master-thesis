

# File to train a model from scratch

import argparse
import pytorch_lightning as pl
import torch
import yaml
import numpy as np
from datasets import Dataset
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader

from custom_datasets.BayesRiskDataset.BayesRiskDatasetLoader import BayesRiskDatasetLoader


from models.estimation.ReferenceModels.factory import ReferenceModelFactory


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str,
                        default='./configs/estimation/unigram_f1/reference-model.yml',
                        help='config to load model from')



    parser.add_argument('--on-hpc', dest='on_hpc', action="store_true",
                        help='Set to true if we are on a hpc')

    parser.set_defaults(on_hpc=False)

    args = parser.parse_args()

    pl.seed_everything(1)
    np.random.seed(1)

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    factory = ReferenceModelFactory(config["model_config"])

    model = factory.create_model()

    dataset_config = config["dataset"]

    n_hypotheses = dataset_config["n_hypotheses"]
    n_references = dataset_config["n_references"]
    utility = dataset_config["utility"]
    sampling_method = dataset_config["sampling_method"]

    dataset_dir = dataset_config["dir"]
    train_dataset_loader = BayesRiskDatasetLoader("train_predictive", n_hypotheses, n_references,
                                                  sampling_method, utility, develop=False,
                                                  base=dataset_dir)

    validation_dataset_loader = BayesRiskDatasetLoader("validation_predictive", n_hypotheses, n_references,
                                                       sampling_method, utility, develop=False,
                                                       base=dataset_dir)

    train_dataset = train_dataset_loader.load(type="pandas").data
    validation_dataset = validation_dataset_loader.load(type="pandas").data

    # We want to transform this thing to a extended dataset. We use pandas for this
    validation_dataset = preprocess_dataset(validation_dataset)
    train_dataset = preprocess_dataset(train_dataset)

    train_dataset = Dataset.from_pandas(train_dataset)

    validation_dataset = Dataset.from_pandas(validation_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=Collate(),
                                  batch_size=config["batch_size"], shuffle=True, )
    val_dataloader = DataLoader(validation_dataset,
                                collate_fn=Collate(),
                                batch_size=config["batch_size"], shuffle=False, )

    # save_loc = config["save_loc"] + '/{}/'.format(utility)
    # path = path_manager.get_abs_path(save_loc)

    # checkpoint_callback = CheckpointCallback(pl_factory, path)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min",
                                        check_finite=True,
                                        divergence_threshold=3)

    trainer = pl.Trainer(
        max_epochs=20,
        gpus=1,
        progress_bar_refresh_rate=1,
        val_check_interval=0.5,
        callbacks=[LearningRateMonitor(logging_interval="step"), early_stop_callback]
    )

    # create the dataloaders
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, )


if __name__ == '__main__':
    main()
