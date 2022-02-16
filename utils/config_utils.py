import transformers
import yaml
import pandas as pd
from datasets import Dataset
import torch
from transformers import MarianTokenizer, MarianMTModel
import numpy as np
from models.pl_predictive_model import GaussianPredictiveModelPL, MLEPredictiveModelPL
from utils.dataset_utils import get_dataset
from utils.metric_utils import get_sacrebleu
from utils.train_utils import preprocess_tokenize
import ast


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


def parse_predictive_config(config_ref, pretrained=False):
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
    pl_model = MLEPredictiveModelPL(nmt_model, tokenizer)

    result["pl_model"] = pl_model

    # load the dataset
    train_dataset, validation_dataset = get_predictive_dataset(config["dataset"]["name"], )

    result["train_dataset"] = train_dataset
    result["validation_dataset"] = validation_dataset

    preprocess_function = lambda x: preprocess_tokenize(x, tokenizer)
    result["preprocess_function"] = preprocess_function

    return result


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


def load_metrics(config, tokenizer):
    metrics = []
    for metric in config["metrics"]:
        metrics.append(get_metric(metric, tokenizer))
    # Create compute metrics function

    return metrics


def get_metric(metric_config, tokenizer):
    '''
    Gets a metric
    :param metric_config:
    :param tokenizer: Tokenizer to tokenize the output of the model with
    :return:
    '''
    # pass
    if metric_config["name"] == "sacreblue":
        return get_sacrebleu(tokenizer)
    else:
        raise ValueError("Metric not found")


def get_predictive_dataset(name, pandas=True):
    if name != "develop":
        raise NotImplementedError("should implement this function properly")
    validation_dataset = pd.read_csv('./data/validation_predictive_helsinki-tatoeba-de-en_1000_bayes_risk.csv',
                                     sep="\t")
    train_dataset = pd.read_csv('./data/train_predictive_helsinki-tatoeba-de-en_1000_bayes_risk.csv',
                                sep="\t")

    validation_dataset = Dataset.from_pandas(split_columns(validation_dataset))
    train_dataset = Dataset.from_pandas(split_columns(train_dataset))

    return train_dataset, validation_dataset


def split_columns(dataset):
    dataset["utility"] = dataset["utilities"].apply(lambda x: np.mean(ast.literal_eval(x)))

    # dataset = dataset.explode("utilities")
    # dataset.rename(columns={"utilities": "utility"}, inplace=True)

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
        model = GaussianPredictiveModelPL(nmt_model, tokenizer)

        model.linear_layers.load_state_dict(state_dict["linear_layers"])
    elif type == "MSE":
        model = MLEPredictiveModelPL(nmt_model, tokenizer)

        model.linear_layers.load_state_dict(state_dict["linear_layers"])
    else:
        raise ValueError("No model of type: {}".format(type))

    return model
