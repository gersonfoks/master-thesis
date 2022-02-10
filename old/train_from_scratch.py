# File to train a model from scratch
import transformers
from transformers import MarianMTModel, MarianTokenizer
import argparse
import torch
from datasets import load_dataset, load_metric, Dataset
from transformers import MarianMTModel, MarianTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from utils.dataset_utils import get_dataset
from utils.train_utils import preprocess_tokenize, get_compute_metrics
from transformers import MarianModel, MarianConfig


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Finetuning a model')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for train (default: 16)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataset-size', type=int, default=-1, metavar='N',
                        help='input dataset size (for debugging, default:-1 (full size)) ')
    parser.add_argument('--dataset', type=str, default="tatoeba", help="The dataset to create the statistics for")
    parser.add_argument('--grad_acc_steps', type=int, default=4, help="number of steps to accumulate the gradient")
    args = parser.parse_args()



    dataset_splits = get_dataset(args.dataset)

    train_dataset = Dataset.from_dict(dataset_splits["train_finetune"])
    validation_dataset = Dataset.from_dict(dataset_splits["validation"])
    if args.dataset_size > 0:
        train_dataset = Dataset.from_dict(train_dataset[:args.dataset_size])
        validation_dataset = Dataset.from_dict(validation_dataset[:args.dataset_size])

    ### Preprocessing
    model_checkpoint = 'Helsinki-NLP/opus-mt-de-en'
    tokenizer = MarianTokenizer.from_pretrained(model_checkpoint)
    preprocess_function = lambda x: preprocess_tokenize(x, tokenizer)

    train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True, )
    validation_tokenized_dataset = validation_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, )

    configuration = transformers.AutoConfig.from_pretrained(model_checkpoint)

    model = MarianMTModel(configuration)


    training_args = TrainingArguments(
        output_dir='./data/results',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_strategy="epoch",
        seed=args.seed,
        # disable_tqdm=True,
        lr_scheduler_type="constant",
        gradient_accumulation_steps=args.grad_acc_steps

    )
    #compute_metrics = get_compute_metrics(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=validation_tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,

        #compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == '__main__':
    main()
