import os
import sys
import transformers
from transformers import MarianTokenizer, MarianMTModel
from pathlib import Path



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


        config_path = os.path.join(str(Path.home()),'FBR', config["model"]["checkpoint"] )
        print(config_path)
        model = Base.from_pretrained(config_path)
    else:
        configuration = transformers.AutoConfig.from_pretrained(model_name)
        model = Base(configuration)

    return model, tokenizer
