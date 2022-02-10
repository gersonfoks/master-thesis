# File to train a model from scratch

import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader


from utils.config_utils import parse_predictive_config, load_model, save_model
from utils.dataset_utils import get_predictive_collate_fn


def get_preprocess_function(tokenizer):
    def preprocess_function(examples, tokenizer, ):
        source = examples["source"]
        targets = examples["hypothesis"]
        model_inputs = tokenizer(source, truncation=True, )
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, truncation=True, )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return lambda examples: preprocess_function(examples, tokenizer)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str, default='./configs/predictive/helsinki-taboeta-de-en-predictive.yml',
                        help='config to load model from')

    args = parser.parse_args()

    parsed_config = parse_predictive_config(args.config, pretrained=False)
    nmt_model = parsed_config["nmt_model"]
    pl_model = parsed_config["pl_model"]
    tokenizer = parsed_config["tokenizer"]
    train_dataset = parsed_config["train_dataset"]
    val_dataset = parsed_config["validation_dataset"]

    preprocess_function = get_preprocess_function(tokenizer)
    #train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)

    collate_fn = get_predictive_collate_fn(nmt_model, tokenizer, )

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=32, shuffle=True, )
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=32, shuffle=False, )

    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=2, max_epochs=1,)#

    trainer.fit(pl_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, )

    save_model(pl_model, parsed_config["config"], "./data/develop_model")

    loaded_pl_model = load_model("./data/develop_model")

    # Validate new model (just to be sure)
    trainer.validate(loaded_pl_model, val_dataloaders=val_dataloader, )


if __name__ == '__main__':
    main()
