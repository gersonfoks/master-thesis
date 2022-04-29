import argparse
import math
import torch
import ast
import pytorch_lightning as pl
import numpy as np
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from custom_datasets.PreprocessedBayesRiskDatasetLoader import BayesRiskDataset, PreprocessedBayesRiskDatasetLoader
from models.estimation.PLPredictiveModelFactory import PLPredictiveModelFactory
from datetime import datetime


def count_to_mean(counter):
    total_count = 0
    total = 0
    for value, c in counter.items():
        total += value * c
        total_count += c

    return total / total_count


class Collator:

    def __init__(self, ref_table):
        self.ref_table = ref_table

    def __call__(self, batch):
        print(batch)
        return batch


# def collate_fn(batch):
#     print(batch)
#
#     return batch
#     # # First get the features
#     # start_time = datetime.now()
#     # ids = [s["ref_id"] for s in batch]
#     # info = ref_table.take(ids, )
#     #
#     # print('Query takes: {}'.format(datetime.now() - start_time))
#     #
#     # start_time = datetime.now()
#     # utilities = torch.tensor([count_to_mean(ast.literal_eval(s["utilities"])) for s in batch])
#     # print('mapping utilities takes: {}'.format(datetime.now() - start_time))
#     #
#     #
#     # sources = info["sources"].flatten()
#     # hypothesis = info["hypothesis"].flatten()
#     #
#     # start_time = datetime.now()
#     # features = {feature_name: torch.Tensor(np.stack(info[feature_name].to_numpy())) for feature_name in
#     #             pl_model.feature_names}
#     # print('processing features takes: {}'.format(datetime.now() - start_time))
#     #
#     #
#     # return features, (sources, hypothesis), utilities


def train_model_tune(config, model_config, dataset_config, num_epochs=10, num_gpus=1, develop=False, data_dir='~/temp'):
    # First we parse the config

    config = {**config, **model_config}

    tune.get_trial_dir()
    pl_model = PLPredictiveModelFactory.create_model(config, )

    bayes_risk_dataset = PreprocessedBayesRiskDatasetLoader(dataset_config, pl_model, develop=develop)
    #
    datasets = bayes_risk_dataset.load()



    pl_model.set_mode('features')

    train_dataloader = DataLoader(datasets["train_predictive"], collate_fn=Collator(),
                                  batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(datasets["validation_predictive"], collate_fn=Collator(),
                                batch_size=config["batch_size"], shuffle=False, num_workers=4)

    trainer = pl.Trainer(
        max_epochs=num_epochs,

        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "val_loss": "val_loss",
                    "train_loss": "train_loss",
                },
                on="validation_end")
        ]
    )

    # create the dataloaders

    trainer.fit(pl_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, )


def main():
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str, default='./configs/predictive/hyperparam_search_pooled.yml',
                        help='config to load model from')

    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    dataset_config = config["dataset"]

    tune_asha(config, config["model_config"], dataset_config, develop=args.develop)


tune_functions = {
    'loguniform': tune.loguniform,
    'uniform': tune.uniform
}


def get_tune(config):
    return tune_functions[config["type"]](config["values"][0], config["values"][1])


def create_search_space(config):
    # Create predictive layer search space

    layers = {'predictive_layers': tune.choice(
        [config["predictive_layers"][name] for name in config["predictive_layers"].keys()])}

    search_space = {
        'lr': get_tune(config["lr"]),
        'dropout': get_tune(config["dropout"]),
        'activation_function': tune.choice(config["activation_function"]),
        'batch_size': tune.choice(config["batch_size"]),
        **layers

    }

    return search_space


def tune_asha(config, model_config, dataset_config, develop=False, num_samples=10, num_epochs=10, gpus_per_trial=1,
              data_dir="~/temp"):
    # First create the search space
    search_space = create_search_space(config["hyperparams"])

    # Create  the scheduler
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2,

    )
    # Fix the fixed params
    train_fn_with_parameters = tune.with_parameters(train_model_tune,
                                                    model_config=model_config,
                                                    dataset_config=dataset_config,
                                                    develop=develop,
                                                    num_epochs=num_epochs,
                                                    num_gpus=gpus_per_trial,
                                                    data_dir=data_dir)

    # Fix the resources
    resources_per_trial = {"cpu": 8, "gpu": gpus_per_trial}

    reporter = CLIReporter(
        parameter_columns=list(search_space.keys()),
        metric_columns=["val_loss", "train_loss", "training_iteration"])

    analysis = tune.run(train_fn_with_parameters,
                        resources_per_trial=resources_per_trial,
                        metric="val_loss",
                        mode="min",
                        config=search_space,
                        num_samples=num_samples,
                        scheduler=scheduler,
                        progress_reporter=reporter,

                        name="tune_mnist_asha")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == '__main__':
    main()

    # Step one is loading the model.
