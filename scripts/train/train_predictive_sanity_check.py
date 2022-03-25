# Train a model with no overfitting countermeasures as a sanity check

import argparse
import pytorch_lightning as pl
import yaml
import numpy as np

from torch.utils.data import DataLoader

from callbacks.CustomCheckpointCallback import CheckpointCallback
from callbacks.predictive_callbacks import MyShuffleCallback
from custom_datasets.FastPreBayesDataset import FastPreBayesDatasetLoader

from models.pl_predictive.PLPredictiveModelFactory import PLPredictiveModelFactory
from scripts.Collate import SequenceCollator, util_functions
from utils.PathManager import get_path_manager


def main():
    pl.seed_everything(1)
    np.random.seed(1)
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str,
                        default='./configs/predictive/tatoeba-de-en-cross-attention-gaussian-mixture-2-sanity-check.yml',
                        help='config to load model from')

    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    parser.add_argument('--on-hpc', dest='on_hpc', action="store_true",
                        help='Set to true if we are on a hpc')

    parser.set_defaults(on_hpc=False)

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    pl_factory = PLPredictiveModelFactory(config['model_config'])
    pl_model = pl_factory.create_model()
    pl_model.set_mode('features')
    dataset_config = config["dataset"]

    preprocess_dir = dataset_config["preprocess_dir"] + "{}_{}/".format(dataset_config["n_hypotheses"],
                                                                        dataset_config["n_references"])

    if args.on_hpc:
        train_max_tables = 10
        val_max_tables = 10
    else:
        train_max_tables = 4
        val_max_tables = 6

    bayes_risk_dataset_loader_train = FastPreBayesDatasetLoader(preprocess_dir, "train_predictive",
                                                                pl_model.feature_names,
                                                                develop=args.develop, max_tables=train_max_tables,
                                                                repeated_indices=dataset_config["repeated_indices"],
                                                                on_hpc=args.on_hpc)
    bayes_risk_dataset_loader_val = FastPreBayesDatasetLoader(preprocess_dir, "validation_predictive",
                                                              pl_model.feature_names,
                                                              develop=args.develop, max_tables=val_max_tables,
                                                              repeated_indices=dataset_config["repeated_indices"],
                                                              on_hpc=args.on_hpc)

    train_dataset = bayes_risk_dataset_loader_train.load()
    val_dataset = bayes_risk_dataset_loader_val.load()

    util_function = util_functions[config['model_config']["loss_function"]]

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=SequenceCollator(pl_model.feature_names, util_function),
                                  batch_size=config["batch_size"], shuffle=False, )
    val_dataloader = DataLoader(val_dataset,
                                collate_fn=SequenceCollator(pl_model.feature_names, util_function),
                                batch_size=config["batch_size"], shuffle=False, )

    path_manager = get_path_manager()

    path = path_manager.get_abs_path('predictive/tatoeba-de-en/models/gaussian_mixture_sanity_check/')

    checkpoint_callback = CheckpointCallback(pl_factory, path, metric="train_loss")



    trainer = pl.Trainer(
        max_epochs=25,
        gpus=1,
        progress_bar_refresh_rate=1,
        val_check_interval=0.5,
        callbacks=[MyShuffleCallback(train_dataset), checkpoint_callback,]
    )

    # create the dataloaders
    trainer.fit(pl_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, )


if __name__ == '__main__':
    main()
