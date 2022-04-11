import argparse
import math
import pytorch_lightning as pl

import yaml
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

from callbacks.CustomCheckpointCallback import CheckpointCallback
from callbacks.predictive_callbacks import MyShuffleCallback
from custom_datasets.FastPreBayesDataset import FastPreBayesDatasetLoader

from models.pl_predictive.PLPredictiveModelFactory import PLPredictiveModelFactory
from scripts.Collate import SequenceCollator, util_functions


def main():
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str, default='./configs/predictive/unigram_f1\hyperparam_search_MSE.yml',
                        help='config to load model from')

    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.add_argument('--on-hpc', dest='on_hpc', action="store_true",
                        help='If true we are on LISA hpc')

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true we are performing a test to see if everything runs as it should')

    parser.add_argument('--name', type=str, default='hyperparam-search',
                        help='name of the hyperparameter search')

    parser.set_defaults(develop=False)
    parser.set_defaults(on_hpc=False)
    parser.set_defaults(smoke_test=False)

    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    dataset_config = config["dataset"]

    gpus_per_trial = 1
    tune_asha(config, config["model_config"], dataset_config, develop=args.develop, on_hpc=args.on_hpc,
              smoke_test=args.smoke_test, gpus_per_trial=gpus_per_trial, name=args.name)


def tune_asha(config, model_config, dataset_config, develop=False, on_hpc=False, smoke_test=False, gpus_per_trial=1,
              name="name"):
    # First create the search space
    search_space = create_search_space(config["hyperparams"])

    # Set numbers to something small to test if everything works.
    if smoke_test:
        grace_period = 1
        max_t = 2
        n_trials = 4
        max_epochs = 2
    else:
        grace_period = config["grace_period"]
        max_t = config["max_epochs"]
        max_epochs = config["max_epochs"]
        n_trials = config["n_trials"]

    scheduler = ASHAScheduler(
        max_t=max_t,
        grace_period=grace_period,
    )
    # Fix the fixed params
    train_fn_with_parameters = tune.with_parameters(train_model_tune,
                                                    model_config=model_config,
                                                    dataset_config=dataset_config,
                                                    develop=develop,
                                                    num_epochs=max_epochs,
                                                    num_gpus=1,
                                                    on_hpc=on_hpc
                                                    )

    # Fix the resources
    resources_per_trial = {"cpu": 3, "gpu": 1}

    reporter = CLIReporter(
        parameter_columns=list(search_space.keys()),
        metric_columns=["val_loss", "train_loss", "training_iteration"])

    analysis = tune.run(train_fn_with_parameters,
                        resources_per_trial=resources_per_trial,
                        metric="val_loss",
                        mode="min",
                        config=search_space,
                        num_samples=n_trials,
                        scheduler=scheduler,
                        progress_reporter=reporter,
                        name=name)

    print("Best hyperparameters found were: ", analysis.best_config)


tune_functions = {
    'loguniform': tune.loguniform,
    'uniform': tune.uniform
}


def create_search_space(config):
    # Create predictive layer search space

    predictive_layers = {'predictive_layers': tune.choice(
        [config["predictive_layers"][name] for name in config["predictive_layers"].keys()])}

    attention_layers = {}
    if "attention_layers" in list(config.keys()):
        attention_layers = {
            k: tune.choice(v) for k, v in config["attention_layers"].items()
        }

    query_layers = {}
    if "query_layers" in list(config.keys()):
        query_layers = {
            k: tune.choice(v) for k, v in config["query_layers"].items()
        }

    search_space = {
        'lr': get_tune(config["lr"]),
        'dropout': get_tune(config["dropout"]),
        'weight_decay': get_tune(config["weight_decay"]),
        'activation_function': tune.choice(config["activation_function"]),
        'batch_size': tune.choice(config["batch_size"]),

        **predictive_layers,
        **attention_layers,
        **query_layers,

    }

    return search_space


def get_tune(config):
    return tune_functions[config["type"]](config["values"][0], config["values"][1])


def combine_hyperparameters_and_config(hyperparams, fixed_model_config):
    query_layers = {
        "query_layers": {"n_heads": hyperparams["n_heads"], "n_queries": hyperparams["n_queries"], }
    }
    attention_layers = {
        "cross_attention_layers": {"n_heads": hyperparams["n_heads"], **fixed_model_config["cross_attention_layers"]}
    }

    result = {**hyperparams, **fixed_model_config, **query_layers, **attention_layers}
    return result


def train_model_tune(hyperparams, model_config, dataset_config, num_epochs=15, num_gpus=1, develop=False, on_hpc=False):
    # First we parse the config

    # Combine the hyperparams with the model config
    model_config = combine_hyperparameters_and_config(hyperparams, model_config)

    pl_model_factory = PLPredictiveModelFactory(model_config, )
    pl_model = pl_model_factory.create_model()

    preprocess_dir = dataset_config["preprocess_dir"] + "{}/{}_{}/".format(dataset_config["utility"], dataset_config["n_hypotheses"],
                                                                        dataset_config["n_references"])

    if on_hpc:
        max_tables = 10
    else:
        max_tables = 6

    bayes_risk_dataset_loader_train = FastPreBayesDatasetLoader(preprocess_dir, "train_predictive",
                                                                pl_model.feature_names,
                                                                develop=develop, max_tables=max_tables,
                                                                repeated_indices=dataset_config["repeated_indices"],
                                                                on_hpc=on_hpc)
    bayes_risk_dataset_loader_val = FastPreBayesDatasetLoader(preprocess_dir, "validation_predictive",
                                                              pl_model.feature_names,
                                                              develop=develop, max_tables=max_tables,
                                                              repeated_indices=dataset_config["repeated_indices"],
                                                              on_hpc=on_hpc)

    train_dataset = bayes_risk_dataset_loader_train.load()
    val_dataset = bayes_risk_dataset_loader_val.load()

    pl_model.set_mode('features')

    util_function = util_functions[model_config["loss_function"]]
    checkpoint_callback = CheckpointCallback(pl_model_factory, tune.get_trial_dir())
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=SequenceCollator(pl_model.feature_names, util_function),
                                  batch_size=hyperparams["batch_size"], shuffle=False, )
    val_dataloader = DataLoader(val_dataset,
                                collate_fn=SequenceCollator(pl_model.feature_names, util_function),
                                batch_size=hyperparams["batch_size"], shuffle=False, )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min",
                                        check_finite=True,
                                        divergence_threshold=5)

    progress_bar_refresh_rate = 0 if on_hpc else 1
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        callbacks=[
            TuneReportCallback(
                {
                    "val_loss": "val_loss",
                    "train_loss": "train_loss",
                },
                on="validation_end"),

            MyShuffleCallback(train_dataset),
            checkpoint_callback,
            early_stop_callback
        ],

        val_check_interval=0.5,
    )

    # Fit the model
    trainer.fit(pl_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, )


if __name__ == '__main__':
    main()

    # Step one is loading the model.
