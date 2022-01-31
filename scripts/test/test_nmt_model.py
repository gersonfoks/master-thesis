### Script for testing a model


import argparse
import torch
from datasets import tqdm, load_metric
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments

from utils.config_utils import parse_config
from utils.dataset_utils import save_dict_to_json
from utils.translation_model_utils import translate


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Finetuning a model')
    parser.add_argument('--config', type=str, default='./configs/example.yml',
                        help='config to load model from')

    args = parser.parse_args()

    # Parse the config and load all the things we need
    parsed_config = parse_config(args.config, test=True)
    model = parsed_config["model"]
    tokenizer = parsed_config["tokenizer"]
    model = model.to("cuda")
    model.eval()
    # Next we preprocess the dataset
    test_dataset = parsed_config["dataset"]["test"].map(parsed_config["preprocess_function"], batched=True)
    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=parsed_config["tokenizer"],
                                           padding=True, return_tensors="pt")

    keys = [
        "input_ids",
        "attention_mask",
        "labels"
    ]

    def collate_fn(batch):

        new_batch = [{k: s[k]for k in keys} for s in batch]
        x_new = data_collator(new_batch)

        sources = [s["translation"][parsed_config["config"]["dataset"]["source"]] for s in batch]
        targets = [s["translation"][parsed_config["config"]["dataset"]["target"]] for s in batch]

        return x_new, (sources, targets)

    eval_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=8)

    # Keep track of intermediate results
    sum_loss = 0
    sum_accs = 0
    total_tokens = 0

    # Evaluate the model
    sacreblue_metric = load_metric('sacrebleu')
    # First accumulate the metrics and then calculate the average.
    with torch.no_grad():
        for x, (sources, targets) in tqdm(eval_dataloader):

            # Get the labels out
            labels = x["labels"].to("cuda")

            non_pad_labels = labels != -100

            n_tokens = torch.sum(non_pad_labels.long()).item()

            # Put to cuda
            x = {k: v.to("cuda") for k, v in x.items()}
            outputs = model.forward(**x)
            sum_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
            # Keep track of the loss

            # Calculate Accuracy

            logits = outputs.logits
            indices = torch.argmax(logits, dim=-1)

            correct = (indices == labels) * non_pad_labels

            sum_accs += torch.sum(correct).item()


            # Calculate Blue by getting a random sample
            translations = translate(model, tokenizer, sources, batch_size=8, )

            targets = [[txt] for txt in targets]
            sacreblue_metric.add_batch(predictions=translations, references=targets)



        #Save the results
        loss = sum_loss/total_tokens
        acc = sum_accs / total_tokens
        bleu = sacreblue_metric.compute()


        test_results = {
            "loss": loss,
            "acc": acc,
            "sacrebleu": bleu
        }

        print("results")
        print(test_results)
        ref = "./results/{}.json".format(parsed_config["config"]["name"])
        save_dict_to_json(test_results, ref)


if __name__ == '__main__':
    main()
