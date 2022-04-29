import ast

import transformers
import yaml
import pandas as pd
from datasets import Dataset
import torch
from transformers import MarianTokenizer, MarianMTModel

from models.estimation.GaussianPredictiveModel import GaussianPredictiveModel
from models.estimation.MLEPredictiveModel import MSEPredictiveModel

from utils.dataset_utils import get_dataset

from utils.train_utils import preprocess_tokenize

from torch import nn


def parse_config(config_ref, pretrained=False):
    '''
    Parses a config and puts all the things in it such that we can work with it
     - Model
     - Data
     - Trainer
     - etc.
    :param config: Reference to the config
    :param test: Whether we are testing or train (regulates what we will load, e.g. when testing we don't need to load the trainer
    :return:
    '''
    # Load the config to a dict
    config = None
    with open(config_ref, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # Get each part of the config

    # Load the model
    model, tokenizer = load_nmt_model(config, pretrained=pretrained)
    result = {
        "config": config,
        "model": model,
        "tokenizer": tokenizer
    }

    # load the dataset
    dataset = get_dataset(config["dataset"]["name"], source=config["dataset"]["source"],
                          target=config["dataset"]["target"])

    result["dataset"] = dataset

    preprocess_function = lambda x: preprocess_tokenize(x, tokenizer)
    result["preprocess_function"] = preprocess_function

    return result


def parse_predictive_config(config_ref, pretrained=False, preprocessing=False):
    '''
    Parses a config and puts all the things in it such that we can work with it
     - Model
     - Data
     - Trainer
     - etc.
    :param config: Reference to the config
    :param test: Whether we are testing or train (regulates what we will load, e.g. when testing we don't need to load the trainer
    :return:
    '''
    # Load the config to a dict
    config = None
    with open(config_ref, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # Get each part of the config

    # Load the model
    nmt_model, tokenizer = load_nmt_model(config["nmt_model"], pretrained=True)
    result = {
        "config": config,
        "nmt_model": nmt_model,
        "tokenizer": tokenizer
    }

    # Load the pl model

    predictive_layers = get_predictive_layers(config["predictive_model"])

    pl_model = None
    if config["predictive_model"]["type"] == "MSE":

        pl_model = MSEPredictiveModel(nmt_model, tokenizer, predictive_layers)
    elif config["predictive_model"]["type"] == "gaussian":
        pass
    else:
        raise ValueError("Type of model unknown: {}, should be one of ['MLE', 'gaussian']".format(type))

    result["pl_model"] = pl_model
    # load the dataset

    if not preprocessing:
        train_dataset, validation_dataset = get_predictive_dataset(config["dataset_train"],
                                                                   config["dataset_validation"])
    else:
        train_dataset, validation_dataset = get_predictive_dataset(config["original_dataset_train"],
                                                                   config["original_dataset_validation"], explode=False)

    result["train_dataset"] = train_dataset
    result["validation_dataset"] = validation_dataset

    preprocess_function = lambda x: preprocess_tokenize(x, tokenizer)
    result["preprocess_function"] = preprocess_function

    return result


activation_functions = {
    'silu': torch.nn.SiLU
}


def get_predictive_layers(architecture_config):
    activation_function = activation_functions[architecture_config['activation_function']]

    layers = []
    # Add all the layers except the last one
    for layer_in, layer_out in zip(architecture_config['layers'][:-2], architecture_config['layers'][1:-1]):
        layers.append(nn.Linear(layer_in, layer_out))
        layers.append(activation_function())
        if architecture_config["dropout"] > 0:
            layers.append(nn.Dropout(p=0.25))

    # Add the last one

    layers.append(nn.Linear(architecture_config['layers'][-2], architecture_config['layers'][-1]))

    return nn.Sequential(*layers)


def load_nmt_model(config, pretrained=False):
    '''
    Loads the model described in the config,
    :param config:
    :param pretrained: if we load a pretrained model or not
    :return:
    '''
    model_name = config["model"]["name"]
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Load the base model
    Base = None
    if config["model"]["type"] == "MarianMT":
        Base = MarianMTModel
    else:
        raise ValueError("Base model not found: {}".format(config["model"]["type"]))

    model = None
    if pretrained:
        model = Base.from_pretrained(config["model"]["checkpoint"])
    else:
        configuration = transformers.AutoConfig.from_pretrained(model_name)
        model = Base(configuration)

    return model, tokenizer


def get_predictive_dataset(dataset_train_config, dataset_validation_config, explode=True):
    train_dataset = load_dataset(dataset_train_config, explode=explode)
    validation_dataset = load_dataset(dataset_validation_config, explode=explode)

    return train_dataset, validation_dataset


def load_dataset(dataset_config, explode=True):
    dataset = pd.read_csv(
        dataset_config["loc"],
        sep="\t")

    if dataset_config["objective"] == 'mean':
        dataset["utility"] = dataset["utilities"].map(lambda x: count_to_mean(ast.literal_eval(x)))

    if explode:
        dataset = dataset.reindex(dataset.index.repeat(dataset["count"])).reset_index()

    # Also collapse based on count.
    return Dataset.from_pandas(dataset)


def count_to_mean(counter):
    total_count = 0
    total = 0
    for value, c in counter.items():
        total += value * c
        total_count += c
    # print(total_count / total)

    return total / total_count


def split_columns(dataset):
    dataset["utility"] = dataset["utilities"]

    return dataset


def save_model(model, config, location, ):
    '''
    Saves the model to the given location
    :param location:
    :return:
    '''
    parameters = {
        "linear_layers": model.linear_layers.state_dict(),
        "config": config,
    }

    torch.save(parameters, location)


def load_model(location, type="MSE"):
    state_dict = torch.load(location)

    nmt_model, tokenizer = load_nmt_model(state_dict["config"]["nmt_model"], pretrained=True)

    if type == "gaussian":
        model = GaussianPredictiveModel(nmt_model, tokenizer)

        model.linear_layers.load_state_dict(state_dict["linear_layers"])
    elif type == "MSE":
        predictive_layers = nn.Sequential(nn.Linear(512 * 2, 512), torch.nn.SiLU(), nn.Dropout(p=0.25),
                                          nn.Linear(512, 1))
        model = MSEPredictiveModel(nmt_model, tokenizer, predictive_layers)

        model.linear_layers.load_state_dict(state_dict["linear_layers"])
    else:
        raise ValueError("No model of type: {}".format(type))

    return model
