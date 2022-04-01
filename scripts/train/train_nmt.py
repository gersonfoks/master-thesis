# File to train a model from scratch

import argparse
import math

from datasets import Dataset
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LambdaLR

from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, \
    get_scheduler
from transformers import DataCollatorForSeq2Seq

from utils.config_utils import parse_config


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str, default='./configs/example.yml',
                        help='config to load model from')

    args = parser.parse_args()

    parsed_config = parse_config(args.config, pretrained=False)
    model = parsed_config["model"]
    tokenizer = parsed_config["tokenizer"]

    # Config variables
    trainer_args = parsed_config["config"]["trainer_args"]

    dataset_splits = parsed_config["dataset"]

    train_dataset = dataset_splits["train_MLE"]
    validation_dataset = dataset_splits["validation_MLE"]

    preprocess_function = parsed_config["preprocess_function"]

    train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True, )
    validation_tokenized_dataset = validation_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, )

    # Use the
    if trainer_args["lr_scheduler_type"] == "inv_sqrt":
        num_warmup_steps = trainer_args["warmup_steps"]

        optimizer = Adam(model.parameters(), lr=trainer_args["learning_rate"], betas=trainer_args["optimizer"]["betas"],
                         eps=float(trainer_args["optimizer"]["eps"]))

        dataset_size = len(train_tokenized_dataset)
        effective_step_size = trainer_args["batch_size"] * trainer_args["gradient_accumulation_steps"]

        steps_per_epoch = math.ceil(dataset_size / effective_step_size)

        # When to start the decay
        start_step_decay = steps_per_epoch * trainer_args["start_decay"] + num_warmup_steps

        def lr_lambda(current_step: int):

            if current_step <= num_warmup_steps:
                return current_step / num_warmup_steps
            # Waiting a number of steps before decaying
            elif current_step <= start_step_decay:
                return 1.0
            else:
                return (current_step - start_step_decay) ** (-0.5)

        optimizers = (optimizer, LambdaLR(optimizer, lr_lambda))

        lr_scheduler_type = "constant"
    else:

        lr_scheduler_type = trainer_args["lr_scheduler_type"]
        optimizers = None

    training_args = Seq2SeqTrainingArguments(
        output_dir='./data/results',

        evaluation_strategy=trainer_args["evaluation_strategy"],
        eval_steps=trainer_args["eval_steps"],
        save_strategy=trainer_args["save_strategy"],
        save_steps=trainer_args["save_steps"],

        learning_rate=trainer_args["learning_rate"],

        per_device_train_batch_size=trainer_args["batch_size"],
        per_device_eval_batch_size=trainer_args["batch_size"],
        gradient_accumulation_steps=trainer_args["gradient_accumulation_steps"],

        num_train_epochs=trainer_args["num_train_epochs"],
        seed=trainer_args["seed"],
        # disable_tqdm=True,

        max_grad_norm=trainer_args["max_grad_norm"],
        metric_for_best_model=trainer_args["metric_for_best_model"],
        group_by_length=True,
        save_total_limit=15,

        # Early stopping
        load_best_model_at_end=trainer_args["load_best_model_at_end"],

        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=trainer_args["warmup_steps"],

    )
    # compute_metrics = get_compute_metrics(tokenizer)

    # Custom trainer uses both inv_sqrt and adam.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=validation_tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=trainer_args["early_stopping_patience"])],
        optimizers=optimizers
    )

    trainer.train()


if __name__ == '__main__':
    main()
