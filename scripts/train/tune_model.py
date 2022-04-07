# File to train a model from scratch

import argparse
import pytorch_lightning as pl
import yaml
import numpy as np
from datasets import Dataset

from pytorch_lightning.callbacks import LearningRateMonitor

from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from custom_datasets.BayesRiskDatasetLoader import BayesRiskDatasetLoader

from models.prompt_tuning.prompt_model_factory import PromptModelFactory

import torch


class DataCollator:

    def __init__(self, nmt_model, tokenizer):
        self.nmt_model = nmt_model
        self.tokenizer = tokenizer

    def __call__(self, batch):
        sources = [b["source"] for b in batch]
        hypotheses = [b["hypotheses"] for b in batch]
        utilities = torch.Tensor(
            [np.sum(np.array(b["utilities"]) * np.array(b["utilities_count"])) / np.sum(np.array(b["utilities_count"]))
             for b in batch])

        model_inputs = self.tokenizer(sources, truncation=True, )
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(hypotheses, truncation=True, )

        model_inputs["labels"] = labels["input_ids"]

        x = [{"labels": l, "input_ids": i, "attention_mask": a} for (l, i, a) in
             zip(model_inputs["labels"], model_inputs["input_ids"], model_inputs["attention_mask"])]

        data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                               padding=True, return_tensors="pt")

        x = data_collator(x).to("cuda")

        return x, (sources, hypotheses), utilities



def main():
    pl.seed_everything(1)
    np.random.seed(1)
    # Training settings
    parser = argparse.ArgumentParser(
        description='Perform prompt tuning')
    parser.add_argument('--config', type=str,
                        default='./configs/predictive/tatoeba-de-en-prompt-tuning.yml',
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

    ### Load the dataset
    dataset_config = config["dataset"]
    train_dataset_loader = BayesRiskDatasetLoader("train_predictive", dataset_config["n_hypotheses"],
                                                  dataset_config["n_references"], dataset_config["sampling_method"],
                                                  args.develop)

    train_dataset = train_dataset_loader.load_as_huggingface_dataset()

    validation_dataset_loader = BayesRiskDatasetLoader("validation_predictive", dataset_config["n_hypotheses"],
                                                       dataset_config["n_references"],
                                                       dataset_config["sampling_method"],
                                                       args.develop)

    validation_dataset = validation_dataset_loader.load_as_huggingface_dataset()

    # We want to "explode the dataset"
    df = train_dataset.to_pandas()

    df = df.explode(["hypotheses", "utilities"]).reset_index(drop=True)

    train_dataset = Dataset.from_pandas(df)

    df = validation_dataset.to_pandas()

    df = df.explode(["hypotheses", "utilities"]).reset_index(drop=True)

    validation_dataset = Dataset.from_pandas(df)

    ### Load the prompt tuning model
    prompt_model_factory = PromptModelFactory(config["model_config"])
    prompt_model = prompt_model_factory.create_model()

    data_collator = DataCollator(prompt_model.nmt_model, prompt_model.tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                  collate_fn=data_collator)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False,
                                       collate_fn=data_collator)


    trainer = pl.Trainer(
        max_epochs=5,
        gpus=1,
        progress_bar_refresh_rate=1,
        val_check_interval=0.5,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        accumulate_grad_batches=config["accumulate_grad_batches"],

    )
    trainer.fit(prompt_model, train_dataloader=train_dataloader, val_dataloaders=validation_dataloader, )


if __name__ == '__main__':
    main()
