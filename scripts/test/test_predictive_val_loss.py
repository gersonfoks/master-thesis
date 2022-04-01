# File to train a model from scratch

import argparse
import pytorch_lightning as pl
import yaml
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
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
        description='Get the validation loss of a model')

    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.add_argument('--config', type=str,
                        default='./configs/predictive/tatoeba-de-en-cross-attention-gaussian-mixture-3-repeated.yml',
                        help='config to load model from')

    parser.set_defaults(develop=False)

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    dataset_config = config["dataset"]

    path_manager = get_path_manager()
    path = path_manager.get_abs_path(config["save_loc"])
    pl_model, factory = PLPredictiveModelFactory.load(path)
    pl_model.set_mode('features')
    pl_model.eval()

    preprocess_dir = dataset_config["preprocess_dir"] + "{}_{}/".format(dataset_config["n_hypotheses"],
                                                                        dataset_config["n_references"])

    train_max_tables = 4
    val_max_tables = 6

    bayes_risk_dataset_loader_val = FastPreBayesDatasetLoader(preprocess_dir, "validation_predictive",
                                                              pl_model.feature_names,
                                                              develop=args.develop, max_tables=val_max_tables,
                                                              repeated_indices=False,
                                                              on_hpc=False)

    val_dataset = bayes_risk_dataset_loader_val.load()

    util_function = util_functions[config['model_config']["loss_function"]]

    val_dataloader = DataLoader(val_dataset,
                                collate_fn=SequenceCollator(pl_model.feature_names, util_function),
                                batch_size=32, shuffle=False, )

    trainer = pl.Trainer(
        max_epochs=25,
        gpus=1,
        progress_bar_refresh_rate=1,

    )

    # create the dataloaders
    trainer.validate(pl_model, val_dataloaders=val_dataloader, )


if __name__ == '__main__':
    main()
