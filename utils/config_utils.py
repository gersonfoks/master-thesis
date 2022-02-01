import transformers
import yaml
from datasets import load_metric
from transformers import MarianTokenizer, MarianMTModel

from utils.dataset_utils import get_dataset
from utils.metric_utils import get_sacrebleu
from utils.train_utils import preprocess_tokenize


def parse_config(config_ref, pretrained=False):
    '''
    Parses a config and puts all the things in it such that we can work with it
     - Model
     - Data
     - Trainer
     - etc.
    :param config: Reference to the config
    :param test: Whether we are testing or training (regulates what we will load, e.g. when testing we don't need to load the trainer
    :return:
    '''
    # Load the config to a dict
    config = None
    with open(config_ref, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # Get each part of the config

    # Load the model
    model, tokenizer = load_model(config, pretrained=pretrained)
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


def load_model(config, pretrained=False):
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
